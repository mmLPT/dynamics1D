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
		self.x=np.zeros(grid.N,dtype=np.complex_)
		self.p=np.zeros(grid.N,dtype=np.complex_)
	
	def setState(self, state,*args): 
		# Commons physical states are implemented
		if state=="coherent":
			# This gives a coherent state occupying a circled area in x/p 
			# representation (aspect ratio), xratio makes possible to 
			# contract the state in x direction
			xratio=1
			
			if len(args)>0:
				xratio=args[0]
				
			sigma=xratio*np.sqrt(self.grid.h/2.0)
			self.x=np.exp(-self.grid.x**2/(2*sigma**2))
			self.normalize("x")

		elif state=="diracx":
			i0=args[0]
			# Set <x|psi> = delta(x-x[i0])
			self.x=np.zeros(self.grid.N)
			self.x[i0]=1.0
			self.normalize("x")
			
		elif state=="diracp":
			# Set <p|psi> = delta(p-p[i0])
			i0=args[0]
			self.p=np.zeros(self.grid.N,dtype=np.complex_)
			self.p[i0]=1.0
			self.p=self.p*self.grid.phaseshift
			self.normalize("p")	
			
	def normalize(self,xp):
		if xp=="x":
			self.x = self.x/np.sqrt(sum(abs(self.x)**2))
			self.x2p()
		elif xp=="p":
			self.p = self.p/np.sqrt(sum(abs(self.p)**2))
			self.p2x()
		
	def shift(self,xp,d0):
		if xp=="x":
			self.p=self.p*np.exp(-1j*d0*self.grid.p/self.grid.h)
			self.p2x()
		elif xp=="p":
			self.x=self.x*np.exp(-1j*d0*self.grid.x/self.grid.h)
			self.x2p()
		
	# === Switching representation x <-> p =============================
	def p2x(self):
		# <p|psi> -> <x|psi>
		self.x=np.fft.ifft(self.p,norm="ortho")
		
	def x2p(self):
		# <x|psi> -> <p|psi>
		self.p=np.fft.fft(self.x,norm="ortho")
	
	# === Operations on wave function ==================================
	def __add__(self,other): 
		# wf1+wf2 <-> |wf1>+|wf2>
		psix=self.x+other.x
		wf=WaveFunction(self.grid)
		wf.x=psix
		wf.x2p()
		return wf
		
	def __sub__(self,other): 
		# wf1-wf2 <-> |wf1>-|wf2>
		psix=self.x-other.x
		wf=WaveFunction(self.grid)
		wf.x=psix
		wf.x2p()
		return wf
		
	def __rmul__(self,scalar): 
		# a*wf <-> a|wf>
		psix=self.x*scalar
		wf=WaveFunction(self.grid)
		wf.x=psix
		wf.x2p()
		return wf
	
	def __mul__(self,scalar):
		# wf*a <-> a|wf>
		psix=self.x*scalar
		wf=WaveFunction(self.grid)
		wf.x=psix
		wf.x2p()
		return wf
		
	def __truediv__(self,scalar): 
		# wf/a <-> |wf>/a
		psix=self.x/scalar
		wf=WaveFunction(self.grid)
		wf.x=psix
		wf.x2p()
		return wf
	
	def __mod__(self,other): 
		# wf1%wf2 <-> <wf1|wf2>
		return sum(np.conj(self.x)*other.x)
		
	def __floordiv__(self,other): 
		# wf1//wf2 <-> |<wf1|wf2>|^2
		return abs(sum(np.conj(self.x)*other.x))**2
		
	def isSymX(self,sigma=2):
		psix=np.flipud(self.x)+self.x
		if  sum(np.conj(psix)*psix)*self.grid.dx > sigma:
			return True
		else:
			return False		
			
	def getMomentum(self,xp,q):
		# Get sum |<psi|psi>|^2q
		if xp=="x":
			return sum(abs(self.x)**(2*q))
		if xp=="p":
			return sum(abs(self.p)**(2*q))
			
class QuantumState:
	def __init__(self,lattice):
		self.lattice=lattice
		self.n=np.zeros(lattice.N,dtype=np.complex_)
		
	def setState(self,state,*args):
		if state=="dirac":
			i0=args[0]+self.lattice.ncenter
			self.n[i0]=1.0
			self.normalize()
			
	def normalize(self):
		self.n = self.n/np.sqrt(sum(abs(self.n)**2))
	
	# === Operations on quantum states =================================
	def __add__(self,other): 
		# wf1+wf2 <-> |wf1>+|wf2>
		wf=WaveFunction(self.lattice)
		wf.n=self.n+other.n
		return wf
		
	def __sub__(self,other): 
		# wf1-wf2 <-> |wf1>-|wf2>
		wf=WaveFunction(self.lattice)
		wf.n=self.n-other.n
		return wf
		
	def __rmul__(self,scalar): 
		# a*wf <-> a|wf>
		wf=WaveFunction(self.lattice)
		wf.n=self.n*scalar
		return wf
		
	def __mul__(self,scalar):
		# wf*a <-> a|wf>
		wf=WaveFunction(self.lattice)
		wf.n=self.n*scalar
		return wf
		
	def __truediv__(self,scalar): 
		# wf/a <-> |wf>/a
		wf=WaveFunction(self.lattice)
		wf.n=self.n/scalar
		return wf
	
	def __mod__(self,other): 
		# wf1%wf2 <-> <wf1|wf2>
		return sum(np.conj(self.n)*other.n)
		
	def __floordiv__(self,other): 
		# wf1//wf2 <-> |<wf1|wf2>|^2
		return abs(sum(np.conj(self.n)*other.n))**2	

	def getMomentum(self,xp,q):
		# Get sum |<psi|psi>|^2q
		return sum(abs(self.n)**(2*q))
