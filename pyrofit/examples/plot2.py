import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import pylab as plt


plt.figure()
data = np.load("lossgrad_lensing2.npz")
c = data['loss']-data['loss'].min()
plt.ylim([0, 16])
plt.xlim([0.7, 1.2])
plt.scatter(data['entropy/a'], c)
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
