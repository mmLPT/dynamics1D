import numpy as np
# ~ from numpy.linalg import matrix_power
from scipy.linalg import expm

from dynamics1D.quantum.discrete.grid import *
from dynamics1D.quantum.discrete.wavefunction import *

import matplotlib.pyplot as plt

class Operator:
	# This class provides a discrete representation for an x/p acting operator
	def __init__(self, grid):
		self.grid=grid
		self.N=grid.N
		self.h=grid.h
		self.M=[]#np.zeros((self.N,self.N),dtype=np.complex_) # The operator representation
		
class TightBindingHamiltonian(Operator):
	def __init__(self,grid,spectrumfile):
		Operator.__init__(self,grid)

		data=np.load(spectrumfile)
		quasienergies=data['quasienergies'][:,0]
		beta=data['beta'][:,0]*2*np.pi
		data.close()
		
		
		self.dE=np.max(quasienergies)-np.min(quasienergies)
		
		
		
		
		self.Vn=np.fft.rfft(quasienergies)/self.N
		
		# ~ print(np.flip(np.delete(self.Vn,0),axis=0).size)
		self.M=np.zeros((self.N,self.N),dtype=np.complex_)
		
		self.M[0]=np.concatenate((self.Vn,np.conjugate(np.flip(np.delete(self.Vn,0),axis=0))))
		
		for i in range(1,self.N):
			self.M[i]=np.roll(self.M[0],i)
		
		# ~ for i in range(0,self.N):
			# ~ for j in range(i,self.N):
				# ~ if np.abs(i-j)<=int(0.5*(self.N-1)):
					# ~ self.M[i,j]=self.Vn[np.abs(i-j)]
				# ~ else:
					# ~ self.M[i,j]=np.conj(self.Vn[self.N-np.abs(i-j)])
				# ~ self.M[j,i]=np.conj(self.M[i,j])
				
				
		diff=quasienergies-np.roll(quasienergies,-1)
		dbeta=0.5*(beta[1]-beta[0])

		ind= ((np.diff(np.sign(np.diff(np.abs(diff)))) < 0).nonzero()[0]+1)
		beta0=beta[ind]
		W=diff[ind]
		ind=beta0>0
		beta0=beta0[ind]+dbeta
		W=W[ind]
		
		
		
		
		n=np.arange(self.Vn.size-1)+1
		self.Vnth=np.zeros(n.size)
		
		def Vnthfun(n,W,beta0):
			return -W/(2*np.pi*n)*2*np.sin(beta0*n)
		
		for i in range(0,W.size):
			self.Vnth+=Vnthfun(n,W[i],beta0[i])
		
		

		ax=plt.gca()
		ax.set_yscale('log')
		ax.set_xlim(0,300)
		ax.set_ylim(10**(-8),10**(-3))
		ax.set_xlabel(r"Distance between sites $n$")
		ax.set_ylabel(r"Coupling $V_n$")
		
		# ~ ax.plot(n,np.abs(Vn),c="blue",zorder=0)
		ax.plot(n,np.abs(self.Vnth),c="red",zorder=1)
		
		plt.show()
				
				
				
				
				
				
				
				
	
class TimePropagator(Operator):
	def __init__(self,grid,T0,hamiltonian):
		Operator.__init__(self,grid)
		self.T0=T0
		self.U=expm(-1j*hamiltonian.M*self.T0/self.h)
	
	def propagate(self,wf):
		wf.n=np.matmul(self.U,wf.n)
				

