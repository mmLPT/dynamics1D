import numpy as np
import random

# This script contains : 1 class
# + class : WaveFunction

class WaveFunction:
	# The wavefunction class is used to describe a state in a 1D 
	# infinite Hiblert space. It provides x and p representation.
	# One needs to switch by hand representation every time it's needed
	
	def __init__(self,grid):
		self.grid=grid
		self.n=np.zeros(grid.N,dtype=np.complex_)
			
	def normalize(self,xp):
		self.n = self.n/np.sqrt(sum(abs(self.n)**2))
	
	# === Operations on wave function ==================================
	def __add__(self,other): 
		# wf1+wf2 <-> |wf1>+|wf2>
		wf=WaveFunction(self.grid)
		wf.n=self.n+other.n
		return wf
		
	def __sub__(self,other): 
		# wf1-wf2 <-> |wf1>-|wf2>
		wf=WaveFunction(self.grid)
		wf.n=self.n-other.n
		return wf
		
	def __rmul__(self,scalar): 
		# a*wf <-> a|wf>
		wf=WaveFunction(self.grid)
		wf.n=self.n*scalar
		return wf
		
	def __mul__(self,scalar):
		# wf*a <-> a|wf>
		wf=WaveFunction(self.grid)
		wf.n=self.n*scalar
		return wf
		
	def __truediv__(self,scalar): 
		# wf/a <-> |wf>/a
		wf=WaveFunction(self.grid)
		wf.n=self.n/scalar
		return wf
	
	def __mod__(self,other): 
		# wf1%wf2 <-> <wf1|wf2>
		return sum(np.conj(self.x)*other.x)
		
	def __floordiv__(self,other): 
		# wf1//wf2 <-> |<wf1|wf2>|^2
		return abs(sum(np.conj(self.x)*other.x))**2	

	def getMomentum(self,xp,q):
		# Get sum |<psi|psi>|^2q
		return sum(abs(self.n)**(2*q))

