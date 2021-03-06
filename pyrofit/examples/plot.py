import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import pylab as plt


plt.figure()
data = np.load("lossgrad_lensing2.npz")
c = data['loss']-data['loss'].min()
#c = np.minimum(25, c)
plt.tricontour(data['entropy/a'], data['entropy/b'], c, levels = [1, 4, 9, 25, 36, 49, 64, 81, 100, 200, 400])
plt.xlabel("a")
plt.ylabel("b")
plt.scatter(data['entropy/a'], data['entropy/b'], c=c)
plt.colorbar()
plt.title(data['loss'].min())
plt.savefig("loss.pdf")

#plt.figure(figsize = (20, 10))
#data = np.load("mock_lensing.npz")
#data2 = np.load("ppd_lensing.npz")
##mu = data['entropy/mu']
#mu2 = data2['entropy/mu']
#obs = data['entropy/obs']
##plt.plot(mu)
#plt.plot(mu2)
#plt.plot(obs, marker='.', ls='')
#plt.savefig("mu.pdf")
