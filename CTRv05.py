#A. Halavanau. DeKalb, IL (2017-2018) / Menlo Park, CA (2018)
#Based on:
#http://inspirehep.net/record/1677406/files/thpmk023.pdf
import numpy as np
import matplotlib.pyplot as plt
from parameters import *
import sys

#Getting angular arguments from the command line (in degrees)
angle1 = str(sys.argv[1])
angle2 = str(sys.argv[2])

theta = float(angle1)*np.pi/180.0  #detector angle theta
phi = float(angle2)*np.pi/180.0  #detector angle theta


def calc_ri(x,y,z,e1,e2,e3,zc):  #obsolete

			 #All coordinates are defined in the beam frame with the origin in the COM
	p0 = ((0,0,zc))  #center of the metallic screen
	p = ((x,y,z))    #particle coordinate	
	e = ((e1,e2,e3)) #direction of particle travel
	R0 = ((0,-Rd,zc)) #detector position

	t = -(np.dot(n,p)-np.dot(n,p0))/np.dot(n,e) # vector to the point of incidence \vec r = \vec p + t * \vec e
	if (t<0): print "Particle won't hit the screen"
 	r = ((x+e1*t,y+e2*t,z+e3*t))  #vector from origin to the point of incidence

        ri = np.sqrt(np.dot(np.subtract(r,p),np.subtract(r,p))) #distance from p to the point of incidence in direction of \vec e 	
	ti = ri/beta/c  #time to reach the screen from point p
	Ri = np.sqrt(np.dot(np.subtract(r,R0),np.subtract(r,R0))) #distance traveled by emitted TR photon


        return (ri,ti,Ri)

def array_sort(tR):
	sort = tR[np.argsort(tR[:, 0])]  #sorting according to the time of arrival
	sort[:,0] = sort[:,0]-sort[0,0]  #first particle arrives at the screen at t_i = 0	
	return sort

def calc_f2(eR,e):

	u = 2.0*np.dot(e,n)/np.dot(n,n)
	e1 = np.subtract(e,np.dot(u,n))  #reflect velocity vector over the screen
	f = np.cross(eR,e)/(1-np.dot(eR,e)) - np.cross(eR,e1)/(1-np.dot(eR,e1))  #f-factor calculation

	return np.dot(f,f)

def calc_tR(dist):       #All coordinates are defined in the beam frame with the origin in the COM

	i = 0
	N = len(dist)    #number of particles
	zc = 10.0	 #dummy constant
	p0 = ((0,0,zc))  #center of the metallic screen
	R0 = ((-Rd*np.cos(theta)*np.sin(phi),-Rd*np.cos(theta)*np.cos(phi),zc+Rd*np.sin(theta))) #detector position

	tR = np.zeros((N,3)) #output array of arrival time and path difference, f^2-factor, where \vec eR = \vec R / |R|

	while i < N:
		x,y,z = dist[i,0], dist[i,1],dist[i,2]
		e1,e2,e3 = dist[i,3],dist[i,4],dist[i,5]
		p = ((x,y,z))    #particle coordinate vector	
		e = ((e1,e2,e3))    #particle velocity vector

		t = -(np.dot(n,p)-np.dot(n,p0))/np.dot(n,e) # vector to the point of incidence \vec r = \vec p + t * \vec e
		if (t<0): print "Particle won't hit the screen"
	 	r = ((x+e1*t,y+e2*t,z+e3*t))  #vector from origin to the point of incidence

     		ri = np.sqrt(np.dot(np.subtract(r,p),np.subtract(r,p))) #distance from p to the point of incidence in direction of \vec e 	
		ti = ri/beta/c  #time to reach the screen from point p
		Ri = np.sqrt(np.dot(np.subtract(R0,r),np.subtract(R0,r))) #distance traveled by emitted TR photon
		
		eR = np.subtract(R0,r) / Ri  #unit vector from point of incidence to the detector
		
		tR[i,0] = ti
		tR[i,1] = Ri
		tR[i,2] = calc_f2(eR,e)
		
		i += 1

        return array_sort(tR)

def calc_F2(nu,tR):
# calculation of coherence factor due to difference in time of arrival and path difference (average f)
	F = np.sum(np.exp(1j*2*np.pi*nu*tR[:,0] + 1j*2*np.pi*nu*tR[:,1]/c))
	F2 = np.abs(F)**2
	return F2

def calc_Sum(nu,tR):
# calculation of coherence factor due to difference in time of arrival and path difference (full 3D calculation)
	fF = np.sum(np.sqrt(tR[:,2])*np.exp(1j*2*np.pi*nu*tR[:,0] + 1j*2*np.pi*nu*tR[:,1]/c)/tR[:,1])
	Sum = np.abs(fF)**2
	return Sum

def test_distribution(N):

	i = j = 0
	d = 0.1   #test case of 1 mm beamlet spacing 
	dist = np.zeros((N,6))  #distribution contains particle coordinates and velocities
	while i<N:
		dist[i,0] -= i*d #np.random.normal(0,d,1)
		i += 1
	while j<N:
		dist[j,5] = beta
		j += 1   

	return dist

def gaussian_distribution(N):   # gaussian distribution

	j = 0
	dist = np.zeros((N,6))
	mu, sigma = 0, 0.1 # mean and standard deviation
	dist[:,0] = np.random.normal(mu, sigma, N) 
	dist[:,1] = np.random.normal(mu, sigma, N) 
	dist[:,2] = np.random.normal(mu, sigmaZ, N)

	while j<N:
		dist[j,5] = beta
		j += 1   

	return dist

def gaussian_distribution_train(N,Nb):   # gaussian distribution

	d = 0.1
	
	dist = np.zeros((N,6))
	mu, sigma = 0, 0.1 # mean and standard deviation
	dist[:,0] = np.random.normal(mu, sigma, N) 
	dist[:,1] = np.random.normal(mu, sigma, N) 
#	dist[:,2] = np.random.normal(mu, sigmaZ, N)
	for i in range (0,Nb): #loop over beamlets
#	    np.random.seed()
	    for j in range (0,N/Nb):  #loop over particles within beamlet
	    	dist[i*N/Nb+j,2] =  np.random.normal(d*i,sigmaZ) #bz[j]	

	j = 0
	while j<N:
		dist[j,5] = beta
		j += 1   

	return dist



def uniform_distribution(N):   # uniform distribution

	i = j = 0
	sigma = 0.1 
	dist = np.zeros((N,6))

	dist[:,0] = np.random.uniform(-3*sigma, 3*sigma, N)
	dist[:,1] = np.random.uniform(-3*sigma, 3*sigma, N)
	dist[:,2] = np.random.uniform(-3*sigmaZ, 3*sigmaZ, N)

	while j<N:
		dist[j,5] = beta
		j += 1   

	return dist

def beamlet_distribution(N,Nb):

	i = j = 0
	d1 = 0.3 #beamlet spacing
	d2 = 0.1 #bunch train separation
	dist = np.zeros((N,6))

	bz = np.random.normal(0.0, sigmaZ, N/Nb)

	for i in range (0,Nb): #loop over beamlets
#	    np.random.seed()
	    for j in range (0,N/Nb):  #loop over particles within beamlet
	        dist[i*N/Nb+j,1] -= d1 * i
	    	dist[i*N/Nb+j,2] =  np.random.normal(d2*i, sigmaZ)   # d2*i   #bz[j]	

#	dist[:,2] = np.random.normal(0.0, sigmaZ, N)
	dist[:,5] = beta
	
	return dist


#distr = gaussian_distribution(8000)
distr = gaussian_distribution_train(8000,8)
#distr = test_distribution(8)
#distr = beamlet_distribution(8000,8)

tR = calc_tR(distr)


#plt.figure()
#plt.plot(distr[:,0],distr[:,1],'.')
#plt.show()

np.savetxt('distr',distr)

i = 0
nu = np.linspace(freq_lower,freq_upper,M)
Sum = np.zeros((M))
dW = np.zeros((M))


while i<M:
	Sum[i] = calc_Sum(nu[i],tR)
	i += 1 

dW = Sum*Rd*Rd/c/4/np.pi/np.pi
np.savetxt('dW',np.c_[nu,dW])




