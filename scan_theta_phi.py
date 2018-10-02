import numpy as np
import os
from parameters import *

theta = theta0 = angle_theta0
phi = phi0 = angle_phi0

deltax = 2*np.abs(angle_theta0) / (Nscan_theta*1.0)
deltay = 2*np.abs(angle_phi0) / (Nscan_phi*1.0)

spectrum = np.zeros(((Nscan_theta+1)*(Nscan_phi+1),3))

for i in range(0,Nscan_theta+1):
    for j in range(0,Nscan_phi+1):
	print theta, phi 
	os.system("python CTRv05.py "+str(theta)+ " "+str(phi))
	datadW = np.loadtxt('dW')
	spectrum[i*(Nscan_theta+1) + j,0] = theta
	spectrum[i*(Nscan_theta+1) + j,1] = phi
	spectrum[i*(Nscan_theta+1) + j,2] = datadW[freq_cut_ind,1]
        phi = phi0 + (j+1)*deltay
    theta = theta0 + (i+1)*deltax
    phi = phi0

np.savetxt('scan_theta_phi',spectrum)
