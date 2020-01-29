# A list of general utility functions to simplify model bulding with pytorch/pyro

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.distributions.transformed_distribution import (
    TransformedDistribution)
from torch.distributions.transforms import ExpTransform
import pyro
from pyro import distributions as dist

def onehot3d(x, weights = None, shape = torch.Size()):
    r"""Returns hot 3-dim tensor.
    
    Points outside of the output boundaries will be clamped to the boundaries.
    
    Arguments:
        x (Tensor): List of points with shape (N, 3), in format (x, y, z)
                    x in [0, W-1]
                    y in [0, H-1]
                    z in [0, D-1]
                    otherwise parameters are clipped at boundaries
        shape (torch.Size): shape in (D, H, W).
        weights (Tensor): Optional weights (default None)
        
    Returns:
        M (Tensor): Output tensor with shape `shape`.
    """
    assert len(shape) == 3, "Shape must be (D, H, W)."
    if shape[0] == 1:
        return onehot2d(x[:,1:], torch.Size([shape[1], shape[2]])).unsqueeze(0)
    device = 'cpu' if not x.is_cuda else x.get_device()
    if weights is None:
        weights = torch.ones(x.shape[0], device = device)
    N = torch.tensor(shape).to(device)
    M = torch.zeros(shape, device = device)
    x = torch.where(x<torch.zeros(1, device = device), torch.zeros(1, device = device), torch.where(x >= N.float()-1, N.float()-1.001, x))
    i = x.long()  # Obtain index tensor
    w = x-i.float()  # Obtain linear weights
    ifl = i[:,0]*N[2]*N[1]+i[:,1]*N[2]+i[:,2]
    M.put_(ifl                 , weights*(1-w[:,0])*(1-w[:,1])*(1-w[:,2]), accumulate = True)
    M.put_(ifl+N[2]*N[1]       , weights*(  w[:,0])*(1-w[:,1])*(1-w[:,2]), accumulate = True)
    M.put_(ifl+N[1]            , weights*(1-w[:,0])*(  w[:,1])*(1-w[:,2]), accumulate = True)
    M.put_(ifl+N[2]*N[1]+N[1]  , weights*(  w[:,0])*(  w[:,1])*(1-w[:,2]), accumulate = True)
    M.put_(ifl               +1, weights*(1-w[:,0])*(1-w[:,1])*(  w[:,2]), accumulate = True)
    M.put_(ifl+N[2]*N[1]     +1, weights*(  w[:,0])*(1-w[:,1])*(  w[:,2]), accumulate = True)
    M.put_(ifl+N[1]          +1, weights*(1-w[:,0])*(  w[:,1])*(  w[:,2]), accumulate = True)
    M.put_(ifl+N[2]*N[1]+N[1]+1, weights*(  w[:,0])*(  w[:,1])*(  w[:,2]), accumulate = True)
    return M

def onehot2d(x, shape = torch.Size()):
    r"""Returns hot 2-dim tensor.
    
    Points outside of the output boundaries will be clamped to the boundaries.
    
    Arguments:
        x (Tensor): List of points with shape (N, 2).
        
    Returns:
        M (Tensor): Output tensor with shape `shape`.
    """
    assert len(shape) == 2, "Shape must be (H, W)."
    N = torch.tensor(shape)
    M = torch.zeros(shape)
    x = torch.where(x<torch.zeros(1), torch.zeros(1), torch.where(x >= N.float()-1, N.float()-1.001, x))
    i = x.long()  # Obtain index tensor
    w = x-i.float()  # Obtain linear weights
    ifl = i[:,0]*N[1]+i[:,1]
    M.put_(ifl       , (1-w[:,0])*(1-w[:,1]), accumulate = True)
    M.put_(ifl+N[1]  , (  w[:,0])*(1-w[:,1]), accumulate = True)
    M.put_(ifl+1     , (1-w[:,0])*(  w[:,1]), accumulate = True)
    M.put_(ifl+N[1]+1, (  w[:,0])*(  w[:,1]), accumulate = True)
    return M

class OnehotConv2d:
    def __init__(self, kernel, shape, method = 'conv2d', flip_kernel = True):
        r"""Convolves hot lists with kernels.
        
        Arguments:
            kernel (Tensor): Convolution kernel with shape (C, kH, kW).
            shape (torch.Size): Ouput image shape (H, W)
            method (string): 'conv2d' or custom 'fft'-based implementation
            flip_kernel (bool): Flip kernel in x and y direction before convolving (default True).
        """
        if kernel.is_cuda:
            device = kernel.get_device()
        else:
            device = 'cpu'

        # Image shape
        assert len(shape) == 2, "shape must be (H, W)."

        # Kernel shape and properties
        assert len(kernel.shape) == 3, "shape must be (C, kH, kW)."
        assert (kernel.shape[1] == kernel.shape[2]), "Kernel must be symmetric"
        if flip_kernel:
            kernel = torch.tensor(np.array(kernel.to('cpu'))[..., ::-1,
                ::-1].copy()).to(device)
        self.odd = (kernel[0].shape[0] % 2 == 1)
        self.kernel = kernel

        # shape  padding
        # 1x1    0
        # 2x2    1
        # 3x3    1
        # 4x4    2
        # ...
        self.padding = int(self.kernel[0].shape[0]/2)

        # Internal shapes
        self.o = 0 if self.odd else 1
        self.hotshape = torch.Size((kernel.shape[0], shape[0]+self.o, shape[1]+self.o))
        self.hotshape_padded = torch.Size((self.hotshape[0],
                              shape[0] + self.padding*2,
                              shape[1] + self.padding*2))

        if method == 'conv2d':
            self._my_call = self._eval_conv2d
        elif method == 'fft':
            self.conv2dfft = ConvKernel2dFFT(self.kernel, self.hotshape_padded,
                    device = device)
            #print("FFT convolution shape:", self.hotshape_padded)
            self._my_call = self._eval_conv2dfft

    def __call__(self, v, weights = None):
        r"""
        Arguments:
            x (Tensor): Hotlist of shape (Npoints, 3), defining (x, y, z, w) coordinates.
        Returns:
            M (Tensor): convolved 2-dim output tensor of shape (H, W).

        Note: If kernel even sized, x and y run from -0.5 to H-0.5 and -0.5 to W-0.5,
        otherwise from 0 to H-1 and 0 to W-1. Values outside this range are clamped.
        """
        x = torch.stack([v[:,2], v[:,1], v[:,0]], dim = 1)
        if not self.odd:
            x[:,1] += 0.5
            x[:,2] += 0.5
        M = onehot3d(x, weights = weights, shape = self.hotshape)
        M_padded = func.pad(M, (self.padding-self.o, self.padding,
                                self.padding-self.o, self.padding))
        #print(M_padded.shape)
        R = self._my_call(M_padded)
        if self.odd:
            return R
        else:
            return R[:-1, :-1]

    def _eval_conv2d(self, M):
        return func.conv2d(M.unsqueeze(0), self.kernel.unsqueeze(0)).squeeze(0).sum(0)
    
    def _eval_conv2dfft(self, M):
        return self.conv2dfft(M).sum(0)

def get_random_module(name, nn_module, all_priors):
    """Puts priors on LensingSystem's parameters. Extends `pyro.random_module()`.
    Notes
    -----
    Might be possible to do this more cleanly...

    Parameters
    ----------
    nn_module : `nn.Module`
        An `nn.Module`.
    all_priors : `dict`
        Must contain a pyro distribution or `torch.tensor`
        for each of `nn_module`'s named parameters that
        has `requires_grad == True`.

    Returns
    -------
    A lifted version of the lensing instance, with parameters
    sampled from the priors. If a prior is `dist.Delta`, the
    parameter is replaced by the value from that distribution
    to ensure compatibility with MCMC functions.
    """
    non_delta_priors = {}
    for p, val in nn_module.named_parameters():
        if val.requires_grad:
            if isinstance(all_priors[p], dist.Delta):
                val.data = all_priors[p].v
            elif isinstance(all_priors[p], torch.Tensor):
                val.data = all_priors[p]
            else:
                non_delta_priors[p] = all_priors[p]

    return pyro.random_module(name, nn_module, non_delta_priors)


def fits_to_torch(ff):
    """Unpacks a FITS file into a `torch.tensor`.

    Returns
    -------
    image : 2D torch.tensor
    """
    # Extract image, being careful to convert to native byte order using
    # the recipe from
    # http://pandas.pydata.org/pandas-docs/version/0.19.1/gotchas.html
    # The orientation is also different in the FITS file
    image = torch.tensor(ff[0].data.byteswap().newbyteorder()).t()
    return image


class TruncatedNormal(dist.Rejector):
    arg_constraints = {"loc": torch.distributions.constraints.real,
                       "scale": torch.distributions.constraints.positive,
                       "x_min": torch.distributions.constraints.dependent,
                       "x_max": torch.distributions.constraints.dependent}

    def __init__(self, loc, scale, x_min=None, x_max=None):
        if x_min is None:
            x_min = -np.inf
        if x_max is None:
            x_max = np.inf

        # Proposal distribution
        propose = dist.Normal(loc, scale)
        # Log of acceptance probability
        def log_prob_accept(x):
            return (x_max > x > x_min).type_as(x).log()
        # Total log probability of acceptance
        log_scale = torch.log(dist.Normal(loc, scale).cdf(x_max) -
                              dist.Normal(loc, scale).cdf(x_min))

        super(TruncatedNormal, self).__init__(propose, log_prob_accept, log_scale)

    def __call__(self, sample_shape=torch.Size([])):
        return self.sample(sample_shape=sample_shape)


class LogUniform(TransformedDistribution):
    def __init__(self, log10_low, log10_high):
        base_dist = torch.distributions.Uniform(np.log(10) * log10_low,
                                                np.log(10) * log10_high)
        super(LogUniform, self).__init__(base_dist, [ExpTransform()])

    def __call__(self, sample_shape=torch.Size([])):
        return self.sample(sample_shape=sample_shape)

    @property
    def log10_low(self):
        return self.base_dist.low / np.log(10)

    @property
    def log10_high(self):
        return self.base_dist.high / np.log(10)

    @property
    def low(self):
        return np.exp(self.base_dist.low)

    @property
    def high(self):
        return np.exp(self.base_dist.high)


def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip



def acosh(x):
    return torch.log(x + torch.sqrt(x**2 - 1.))


def interp1d(xmin, xmax, n, log_x=False, log_y=False, device=None):
    """Torch compatible linear interpolation decorator on regular grid.

    Generates a wrapper function that tabulates and linearly interpolates
    target function in the specified range.

    Usage examples:

    @interp1d(xmin, xmax, n)
    def f(x):
        return x + x**2 + x**3 + ...

    OR

    interp1d(xmin, xmax, n)(f)

    Parameters
    ----------
        xmin : minimum grid value [float]
        xmax : maximum grid value [float]
        n : number of grid points [int]
        log_x : log interpolation on x-axis [bool]
        log_y : log interpolation on y-axis [bool]
    """
    if log_x:
        xmin, xmax = np.log10(xmin), np.log10(xmax)
    xgrid = torch.linspace(xmin, xmax, n).to(device)
    dx = xgrid[1] - xgrid[0]

    def _interp1d(x, fgrid):
        i = torch.clamp(torch.floor((x-xmin)/dx), 0, n-1)
        j = torch.clamp(torch.floor((x-xmin)/dx)+1, 0, n-1)
        w = torch.clamp((x-xmin)/dx - i, 0., 1.).to(device)
        i = i.type(torch.long).to(device)
        j = j.type(torch.long).to(device)
        fint = fgrid[i]*(1-w) + fgrid[j]*w
        return fint

    def wrapper(f):
        if (log_x, log_y) == (False, False):
            fgrid = f(xgrid).to(device)
            return lambda x: _interp1d(x, fgrid)
        elif (log_x, log_y) == (False, True):
            fgrid = torch.log10(f(xgrid), device=device)
            return lambda x: torch.pow(torch.tensor(10., device=device), _interp1d(x, fgrid))
        elif (log_x, log_y) == (True, False):
            fgrid = f(torch.pow(torch.tensor(10., device=device), xgrid)).to(device)
            return lambda x: _interp1d(torch.log10(x), fgrid)
        else:
            fgrid = torch.log10(f(torch.pow(torch.tensor(10., device=device), xgrid))).to(device)
            return lambda x: torch.pow(torch.tensor(10., device=device), _interp1d(torch.log10(x), fgrid))
    return wrapper


def bilinear_interp(x_new_mg, y_new_mg, x_mg, y_mg, img):
    """Bilinear interpolation.

    Parameters
    ----------
    x_new_mg : 2D torch.tensor
        X meshgrid over which to interpolate
    y_new_mg : 2D torch.tensor
        X meshgrid over which to interpolate
    x_mg : 2D torch.tensor
        X-coordinates for img
    y_mg : 2D torch.tensor
        Y-coordinates for img
    img : 2D torch.tensor
        Image values over the specified x and y meshgrids.

    Returns
    -------
    Interpolated image
    """
    assert x_new_mg.shape == y_new_mg.shape
    assert x_mg.shape == y_mg.shape
    assert x_mg.shape == img.shape
    half_width = 0.5 * (x_mg[-1, 0] - x_mg[0, 0])
    half_height = 0.5 * (y_mg[0, -1] - y_mg[0, 0])
    grid = torch.zeros(torch.Size([1]) + x_new_mg.shape + torch.Size([2]))
    grid[:, :, :, 0] = y_new_mg / half_height
    grid[:, :, :, 1] = x_new_mg / half_width
    return F.grid_sample(img.unsqueeze(0).unsqueeze(0), grid,
                         align_corners=True).squeeze()


class TorchInterp3d:
    def __init__(self, data, xrange, yrange, zrange, device='cpu'):
        self.xrange = xrange
        self.xwidth = xrange[1] - xrange[0]
        self.yrange = yrange
        self.ywidth = yrange[1] - yrange[0]
        self.zrange = zrange
        self.zwidth = zrange[1] - zrange[0]
        #self.data = torch.tensor(data, dtype = torch.float32).unsqueeze(0)
        self.data = data.unsqueeze(0)
        self.device = device
        
    def __call__(self, x, y, z):
        x = 2*(x-self.xrange[0])/self.xwidth - 1
        y = 2*(y-self.yrange[0])/self.ywidth - 1
        z = 2*(z-self.zrange[0])/self.zwidth - 1
        xt = x.unsqueeze(1).unsqueeze(1).expand(len(x), len(y), len(z))
        yt = y.unsqueeze(0).unsqueeze(2).expand(len(x), len(y), len(z))
        zt = z.unsqueeze(0).unsqueeze(1).expand(len(x), len(y), len(z))
        grid = torch.zeros((1, len(x), len(y), len(z), 3)).to(self.device)
        grid[0,:,:,:,2] = xt
        grid[0,:,:,:,1] = yt
        grid[0,:,:,:,0] = zt
        return nn.functional.grid_sample(self.data, grid, align_corners=True)[0]
    

class TorchInterp2d:
    def __init__(self, data, xrange, yrange, device='cpu'):
        self.xrange = xrange
        self.xwidth = xrange[1] - xrange[0]
        self.yrange = yrange
        self.ywidth = yrange[1] - yrange[0]
        #self.data = torch.tensor(data, dtype = torch.float32).unsqueeze(0)
        self.data = data.unsqueeze(0)
        self.device = device
        
    def __call__(self, x, y):
        x = 2*(x-self.xrange[0])/self.xwidth - 1
        y = 2*(y-self.yrange[0])/self.ywidth - 1
        xt = x.unsqueeze(1).expand(len(x), len(y))
        yt = y.unsqueeze(0).expand(len(x), len(y))
        grid = torch.zeros((1, len(x), len(y), 2)).to(self.device)
        grid[0,:,:,0] = xt
        grid[0,:,:,1] = yt
        return nn.functional.grid_sample(self.data, grid, align_corners=True)[0]


class ConvKernel2dFFT:
    def __init__(self, kernel, img_shape, device = 'cpu'):
        """Note: Kernel ordering same as conv2d."""
        self.device = device
        kernel = torch.tensor(np.array(kernel.to('cpu'))[...,::-1,::-1].copy()).to(device)  # Flip kernel, in order to get same behaviour as conv2d
        self._init_fft(kernel, img_shape)
        self.shape_k = kernel.shape

    def _init_fft(self, kernel, img_shape):
        s1 = np.array(kernel.shape, dtype=np.int)
        kernel_s3 = torch.zeros(img_shape).to(self.device)
        kernel_s3[..., :s1[-2],:s1[-1]] = kernel
        self.fft_kernel = torch.rfft(kernel_s3, 2, onesided=False)

    def __call__(self, img, crop = True):
        img_conv = self._conv_fft(img.to(self.device))
        return img_conv[..., self.shape_k[-2]-1:, self.shape_k[-1]-1:]

    def _conv_fft(self, img):
        fft_img = torch.rfft(img, 2, onesided=False)
        fft_prod = torch.zeros_like(fft_img)
        fft_prod[...,0] = fft_img[...,0]*self.fft_kernel[...,0]-fft_img[...,1]*self.fft_kernel[...,1]
        fft_prod[...,1] = fft_img[...,0]*self.fft_kernel[...,1]+fft_img[...,1]*self.fft_kernel[...,0]
        ret = torch.irfft(fft_prod, 2, onesided=False)
        return ret

# Turns out to be *way* slower than native CUDA implementations (accessible by just running torch.nn.conv2d on CUDA)
class ConvKernel2d(nn.Module):
    def __init__(self, kernel, shape, use_conv2d = False, device=None):
        """Fast convolution using pytorch FFT.

        Parameters
        ----------
        kernel : torch.tensor
            Require square 2-dim tensor of size (n, m).
            Note: n and m must be odd for CPU implementation to work properly.
        shape : int 2-tuple 
            Size of input / output image, (n, m).
            Note: n and m must be even for CPU implementation to work properly.
        use_conv2d : bool (default false)
            Use torch.nn.functional.conv2d instead of fft implementation.
        """
        
        super(ConvKernel2d, self).__init__()
        if not (
            isinstance(kernel, torch.Tensor) and 
            kernel.dim() == 2 and
            kernel.shape[0] == kernel.shape[1] and
            kernel.shape[0] % 2 == 1 and
            kernel.shape[1] % 2 == 1 and
            shape[0] % 2 == 0 and
            shape[1] % 2 == 0
        ):
            print("WARNING: Problematic kernel / shape format.")
        kernel = kernel.to(device)
        self.device = device

        self.use_conv2d = use_conv2d
        if self.use_conv2d:
            self._init_conv2d(kernel, shape)
        else:
            self._init_fft(kernel, shape)

    def _init_fft(self, kernel, shape):
        self.s1 = np.array(kernel.shape, dtype=np.int)
        self.s2 = np.array(shape, dtype=np.int)
        self.s3 = tuple(self.s1 + self.s2)
        self.shift = (int((self.s1[0]-1)/2), int((self.s1[1]-1)/2))
        
        kernel_s3 = torch.zeros(self.s3, device=self.device)
        kernel_s3[:self.s1[0],:self.s1[1]] = kernel
        self.fft_kernel = torch.rfft(kernel_s3, 2, onesided=False)

    def _init_conv2d(self, kernel, shape):
        self.padding = tuple(((np.array(kernel.shape)-1)/2).astype(np.int))
        self.kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    def __call__(self, img):
        """Perform convolution.
        
        Parameters
        ----------
        img : (batch, n, m) tensor
            Input image, with shape as defined earlier. Additional batch dimensions are possible.
        """
        if self.use_conv2d:
            return self._conv_conv2d(img)
        else:
            return self._conv_fft(img)

    def _conv_conv2d(self, img):
        ret = nn.functional.conv2d(img.unsqueeze(0).unsqueeze(0), self.kernel, padding = self.padding)
        return ret.squeeze(0).squeeze(0)

    def _conv_fft(self, img):
        img_s3 = torch.zeros(self.s3, device = self.device)
        img_s3[:self.s2[0],:self.s2[1]] = img
        fft_img = torch.rfft(img_s3, 2, onesided=False)
        fft_prod = torch.zeros_like(fft_img)
        fft_prod[...,0] = fft_img[...,0]*self.fft_kernel[...,0]-fft_img[...,1]*self.fft_kernel[...,1]
        fft_prod[...,1] = fft_img[...,0]*self.fft_kernel[...,1]+fft_img[...,1]*self.fft_kernel[...,0]
        ret = torch.irfft(fft_prod, 2, onesided=False)
        return ret[self.shift[0]:self.shift[0]+self.s2[0], self.shift[1]:self.shift[1]+self.s2[1]]

class PowerSpectrum2d:
    def __init__(self, shape, nbins):
        """Calculate binned power spectrum of 2-dim images.
        
        - Returns power integrated/summed over logarithmically spaced k-bins.
        - We adopt the convention that pixel size = unit length.
        - Note that sum(power) = var(img).
        """
        self.shape = shape
        self.nbins = nbins
        self.K = self._get_kgrid_2d(shape[0], shape[1])
        self.kedges = torch.logspace(
                torch.log10(min(self.K[1,0], self.K[0,1])),
                torch.log10(self.K.max())+0.001, nbins+1)
        self.kmeans = (self.kedges[1:] * self.kedges[:-1])**0.5
        
    @staticmethod
    def _get_kgrid_2d(nx, ny):
        """Generate k-space grid."""
        kx = torch.linspace(0, 2*np.pi*(1-1/nx), nx)
        kx = (torch.fmod(kx+np.pi, 2*np.pi)-np.pi)*2
        ky = torch.linspace(0, 2*np.pi*(1-1/ny), ny)
        ky = (torch.fmod(ky+np.pi, 2*np.pi)-np.pi)*2
        KX, KY = torch.meshgrid(kx, ky)
        K = (KX**2 + KY**2)**0.5
        return K

    def _P2d(self, img):
        """Calculate power spectrum."""
        # Get variance in fourier space
        fft = torch.rfft(img, signal_ndim = 2, onesided = False)
        A = img.numel()
        var = (fft[:,:,0]**2 + fft[:,:,1]**2)/A**2
        
        # Generate grid output
        out = []
        for i in range(self.nbins):
            s = var[(self.kedges[i]<=self.K)&(self.K<self.kedges[i+1])].sum()
            out.append(s)
        out = torch.stack(out)
        return out
    
    def __call__(self, img):
        return self._P2d(img)


def get_components(yaml_entries, type_mapping, device='cpu'):
    """Instantiates objects defined in the lens plane.

    These may be defined in terms of convergences ('kappas' section in yaml) or
    deflection fields ('alphas' section).

    Parameters
    ----------
    X, Y : torch.Tensor, torch.Tensor
        Lens plane meshgrids.
    entries : dict
        Config entries.
    type_mapping : dict(str -> (Object, dict))
        Mapping from 'type' field in config to the corresponding object and a
        dict containing any extra initializer arguments.
    device : str

    Returns
    -------
    instances : list
        Instances of the relevant classes.
    """
    if yaml_entries is None: return []

    instances = []
    for name, entry in yaml_entries.items():
        entry_type = entry["type"]
        parameters = entry.get("parameters", {})
        options = entry.get("options", {})

        # Parse option values
        for key, val in options.items():
            options[key] = yaml_params2._parse_val(val, device=device)

        sampler = yaml_params2.yaml2sampler(name, parameters, device=device)

        cls, args, kwargs = type_mapping[entry_type]
        kwargs.update(options)
        kwargs.update({"device": device})
        instance = cls(sampler, *args, **kwargs)

        instances.append(instance)
    return instances

def observe(name, value):
    # FIXME: Get device information from value
    device = 'cpu'
    pyro.sample(name, dist.Delta(value, log_density = torch.tensor(0., device = device)), obs = value)


def load_param_store(paramfile, device = 'cpu'):
    """Loads the parameter store from the resume file.
    """
    pyro.clear_param_store()
    try:
        pyro.get_param_store().load(paramfile, map_location = device)
        print("Loading guide:", paramfile)
    except FileNotFoundError:
        print("Could not open %s. Starting with fresh guide."%paramfile)

