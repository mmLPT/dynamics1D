import numpy as np
from numpy.linalg import matrix_power

from dynamics1D.quantum.discrete.grid import *
from dynamics1D.quantum.discrete.wavefunction import *

class Operator:
	# This class provides a discrete representation for an x/p acting operator
	def __init__(self, grid):
		self.grid=grid
		self.N=grid.N
		self.h=grid.h
		self.M=np.zeros((self.N,self.N),dtype=np.complex_) # The operator representation
		
class TightBindingHamiltonian(Operator):
	def __init__(self,grid,spectrumfile):
		Operator.__init__(self,grid)

		data=np.load(spectrumfile)
		quasienergies=data['quasienergies'][:,0]
		beta=data['beta'][:,0]*2*np.pi
		
		# ~ quasienergies=data['qEs'][:,-1]
		# ~ beta=data['beta'][:,-1]*2*np.pi/self.h
		data.close()
		
		self.Vn=np.fft.rfft(quasienergies)/self.N
		
		# ~ for i in range(0,self.N):
			# ~ for j in range(i,self.N):
				# ~ if np.abs(i-j)<=int(0.5*(self.N-1)):
					# ~ self.M[i,j]=self.Vn[np.abs(i-j)]
				# ~ else:
					# ~ self.M[i,j]=np.conj(self.Vn[self.N-np.abs(i-j)])
				# ~ self.M[j,i]=np.conj(self.M[i,j])
				
				
		diff=quasienergies-np.roll(quasienergies,-1)
		dbeta=0.5*(beta[1]-beta[0])
		
		print(1/self.h)
		

		ind= ((np.diff(np.sign(np.diff(np.abs(diff)))) < 0).nonzero()[0]+1)
		beta0=beta[ind]
		W=diff[ind]
		ind=beta0>0
		beta0=beta0[ind]+dbeta
		W=W[ind]
		
		
		def Vnthfun(n,W,beta0):
			return -W/(2*np.pi*n)*2*np.sin(beta0*n)
		
		
		
		Vn=np.delete(self.Vn,0)
		n=np.arange(Vn.size)+1
		Vnth=np.zeros(n.size)
	
		for i in range(0,W.size):
			Vnth+=Vnthfun(n,W[i],beta0[i])
		
		
		
		ax=plt.subplot(1,2,1)
		
		dqE=np.max(quasienergies)-np.min(quasienergies)
		
		ax.plot(beta,quasienergies-np.mean(quasienergies))
		
		ax.scatter(beta0,np.zeros(beta0.size))
		
		ax=plt.subplot(1,2,2)
		
		ax=plt.gca()
		ax.set_yscale('log')
		ax.set_xlim(0,250)
		
		ax.scatter(n,np.abs(Vn),c="blue",s=5,zorder=0)
		ax.scatter(n,np.abs(Vnth),c="red",s=5,zorder=1)
		
		plt.show()
				
				
				
				
				
				
				
				
	
class TimePropagator(Operator):
	def __init__(self,grid,T0):
		Operator.__init__(self,grid,T0,hamiltonian)
		self.T0=T0
		self.U=expm(-1j*hamiltonian.M*self.T0/self.h)
	
	def propagate(self,wf):
		wf=np.matmul(self.U,wf)
				

