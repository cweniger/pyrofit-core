import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import pylab as plt


plt.figure()
data = np.load("lossgrad_lensing.npz")
c = data['loss']-data['loss'].min()
c = np.minimum(25, c)
plt.tricontour(data['entropy/a'], data['entropy/b'], c, levels = [1, 4, 9, 25])
plt.xlabel("a")
plt.ylabel("b")
plt.scatter(data['entropy/a'], data['entropy/b'], c=c)
plt.colorbar()
plt.title(data['loss'].min())
plt.savefig("loss.pdf")

plt.figure()
data = np.load("mock_lensing.npz")
mu = data['entropy/mu']
obs = data['entropy/obs']
plt.plot(mu)
plt.plot(obs, marker='.', ls='')
plt.savefig("mu.pdf")
