import matplotlib.pyplot as plt
import numpy as np
from plotting import *
from parameters import *

data = np.loadtxt('scan_theta_phi')

m1 = Nscan_theta + 1 
m2 = Nscan_phi + 1

xmin = np.min(data[:,0])
xmax = np.max(data[:,0])

ymin = np.min(data[:,1])
ymax = np.max(data[:,1])


values = data[:,2].reshape(m1,m2)
array = np.zeros((m1,m2))

array = values / np.max(values)

plt.figure()
plt.imshow(array, extent=[xmin,xmax,ymin,ymax],aspect='auto')
plt.xlabel(r'$\theta$ (degrees)')
plt.ylabel(r'$\phi$ (degrees)')
cb = plt.colorbar()
cb.set_label('Intensity (arb.)')
plt.tight_layout()
plt.show()
