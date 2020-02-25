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
	
	def setState(self, state,i0=0,xratio=1.0,norm=True): 
		# Commons physical states are implemented
		if state=="coherent":
			# This gives a coherent state occupying a circled area in x/p 
			# representation (aspect ratio), xratio makes possible to 
			# contract the state in x direction
			sigma=xratio*np.sqrt(self.grid.h/2.0)
			self.x=np.exp(-self.grid.x**2/(2*sigma**2))
			self.normalizeX()
			self.x2p()

		elif state=="diracx":
			# Set <x|psi> = delta(x-x[i0])
			self.x=np.zeros(self.grid.N)
			self.x[i0]=1.0
			if norm:
				self.normalizeX()
			self.x2p()
			
		elif state=="diracp":
			# Set <p|psi> = delta(p-p[i0])
			self.p=np.zeros(self.grid.N,dtype=np.complex_)
			self.p[i0]=1.0
			self.p=self.p*self.grid.phaseshift
			self.p2x()
			if norm:
				self.normalizeX()
				self.x2p()
			
			
	def normalizeX(self):
		# Normalize <x|psi>
		nrm=sum(abs(self.x)**2)*self.grid.dx
		self.x = self.x/np.sqrt(nrm)
		
	def normalizeP(self):
		nrm=sum(abs(self.p)**2)*self.grid.dp
		self.p = self.p/np.sqrt(nrm)
		
	def shiftX(self,x0):
		# Shift the wavefunction in x direction
		self.p=self.p*np.exp(-1j*x0*self.grid.p/self.grid.h)
		self.p2x()
		
	def shiftP(self,p0):
		# Shift the wavefunction in x direction
		self.x=self.x*np.exp(1j*p0*self.grid.x/self.grid.h)
		self.x2p()
		
	# === Switching representation x <-> p =============================
	def p2x(self):
		# <p|psi> -> <x|psi>
		# ~ self.x=np.fft.ifft(self.p)*self.grid.N
		self.x=np.fft.ifft(self.p)*self.grid.N*self.grid.dp/np.sqrt(2*np.pi*self.grid.h)
		
	def x2p(self):
		# <x|psi> -> <p|psi>
		# ~ self.p=np.fft.fft(self.x)/self.grid.N
		self.p=np.fft.fft(self.x)*self.grid.dx/np.sqrt(2*np.pi*self.grid.h)
	
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
		return sum(np.conj(self.x)*other.x)*self.grid.dx
		
	def __floordiv__(self,other): 
		# wf1//wf2 <-> |<wf1|wf2>|^2
		return abs(sum(np.conj(self.x)*other.x)*self.grid.dx)**2
		
	# === I/O ==========================================================
	def isSymetricInX(self,sigma=2.0):
		self.p2x()
		
		psix=np.flipud(self.x)+self.x

		if  sum(np.conj(psix)*psix)*self.grid.ddx > sigma:
			return True
		else:
			return False
			
	def isSymetricInP(self,sigma=2.0):
		self.x2p()
		psip=np.flipud(self.p)-self.p

		if  sum(np.conj(psip)*psip)*self.grid.ddp < sigma:
			return True
		else:
			return False

	def getMomentum(self,xp,q):
		# Get sum |<psi|psi>|^2q
		if xp=="x":
			return sum(abs(self.x)**(2*q)*self.grid.ddx)
		if xp=="p":
			return sum(abs(self.p)**(2*q)*self.grid.ddp)