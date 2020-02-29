import numpy as np
import matplotlib.pyplot as plt

# This script contains : 1 class
# + class : grid

class Lattice:
	# The Grid class provides a grid adapted to class WaveFunction
	# Id est, the p array is set to be FFT compatible and well
	# dimenzioned in h. All attributes are properties.
	def __init__(self,N,h):
		self.h=h # hbar value
		self.N=N 
		
		if N%2==0:
			raise ValueError("Number of cells should be odd")
		
		self.n=np.arange(-int(0.5*(self.N-1)),int(0.5*(self.N+1)))
		
		self.ncenter=int(0.5*(self.N-1))
		

class Grid:
	# The Grid class provides a grid adapted to class WaveFunction
	# Id est, the p array is set to be FFT compatible and well
	# dimenzioned in h. All attributes are properties.
	def __init__(self,N,h,xmax=2*np.pi):
		self.h=h # hbar value
		self.N=N 
		self.xmax=xmax 
		
		self.x,self.dx=np.linspace(-xmax/2.0,xmax/2.0,N,endpoint=False,retstep=True)
		self.x=self.x+self.dx/2.0
		
		self.p=np.fft.fftfreq(self.N,self.dx)*self.h*2*np.pi
		self.dp=self.p[1]-self.p[0]
		
		# A p-defined WaveFunction, don't know about x interval, only about 
		# it width, then to center a p-defined wf, you have to multiply
		# by the followinf factor
		self.phaseshift=np.exp(-(1j/self.h)*((self.xmax-self.dx)/2.0)*self.p)
	


	
