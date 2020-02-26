import numpy as np
import matplotlib.pyplot as plt

# This script contains : 1 class
# + class : grid

class Grid:
	# The Grid class provides a grid adapted to class WaveFunction
	# Id est, the p array is set to be FFT compatible and well
	# dimenzioned in h. All attributes are properties.
	def __init__(self,N,h):
		self.h=h # hbar value
		self.N=N 
		
		if N%2==0:
			raise ValueError("Number of cells should be odd")
		
		self.n=np.arange(-int(0.5*(self.N-1)),int(0.5*(self.N+1)))
		
	
	

	
