import numpy as np
import os
from parameters import *

angle0 = angle_theta0
delta = 2*np.abs(angle0) / (Nscan_theta*1.0)


spectrum = np.zeros((M*(Nscan_theta+1),3))
for i in range(0,Nscan_theta+1):
	angle = angle0 + i*delta
	print "Evaluating at angle theta: ", angle
	os.system("python CTRv05.py "+str(angle)+" "+str(0.0))
	datadW = np.loadtxt('dW')
	spectrum[i*M:i*M + M,0] = angle
	spectrum[i*M:i*M + M,1:] = datadW

#file format: angle  frequency  dW
np.savetxt('scan_theta',spectrum)


