import numpy as np
#Simulation parameters
c = 100.0*2.998e+8  #Speed of light in SGS
gamma = 100.0   #mean Lorentz factor of electron beam
beta = np.sqrt(1 - 1/gamma/gamma)  #mean beta factor of electron beam
Rd = 100.0   #Nominal detector location in SGS

n = ((0,-1.0,-1.0))  #normal vector to the metallic plane (assumed plane is at 45 degrees)
		 #plane equation is then y + z = zc, where zc is some constant

M = 250   #frequency domain size
#number of scanning points
Nscan_theta = 75
Nscan_phi = 75

#frequency limits
freq_lower = 1.0e+11
freq_upper = 1.0e+12
freq_cut = 3.0e+11
freq_cut_ind = int(M * freq_cut / (freq_upper - freq_lower))


sigmaZ = 0.01 #0.075 / 5.0  #sigmaZ of the bunch assumed sigma_t = 2.5 ps

#starting scanning point
angle_theta0 = -5.0
angle_phi0 = -5.0
