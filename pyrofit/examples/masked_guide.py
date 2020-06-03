import pyro, pyro.distributions as dist
from pyrofit.core.decorators import Yaml, register


@register
class Model:
    def __init__(self, mask, device=None):
        self.mask = mask

    def __call__(self, x: Yaml, error: Yaml):
        pyro.sample('obs', dist.Normal(x, error).mask(self.mask))


if __name__ == '__main__':
    from operator import itemgetter
    import torch, numpy as np
    from pyro.distributions import transforms
    from matplotlib import pyplot as plt

    obs, mask = itemgetter('obs', 'mask')(np.load('masked_guide.npz'))
    guide = torch.load('masked_guide.guide.pt')

    extract = lambda name: transforms.biject_to(guide['constraints'][name])(guide['params'][name].detach())

    loc = extract('guide_z_loc')
    scale = extract('guide_z_scale')

    X = np.arange(len(loc))
    plt.plot(X[~mask], obs[~mask], 'r.')
    plt.plot(X[mask], obs[mask], 'g.')
    plt.errorbar(X, loc, scale, lw=0.5, elinewidth=1, capsize=2, capthick=1)
