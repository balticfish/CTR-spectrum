import matplotlib.pyplot as plt
import numpy as np
from plotting import *
from parameters import *

data = np.loadtxt('scan_theta')

m1 = Nscan_theta + 1
m2 = M

xmin = np.min(data[:,0])
xmax = np.max(data[:,0])

ymin = np.min(data[:,1]) / 1.0e+12
ymax = np.max(data[:,1]) / 1.0e+12

values = data[:,2].reshape(m1,m2)
array = np.zeros((m1,m2))

array = np.rot90(values) / np.max(values)

plt.figure()
plt.imshow(array, extent=[xmin,xmax,ymin,ymax],aspect='auto')
cb = plt.colorbar()
cb.set_label('Intensity (arb.)')

plt.xlabel(r'$\theta$ (degrees)')
plt.ylabel(r'$\omega/2 \pi$ (THz)')
plt.tight_layout()
plt.show()
