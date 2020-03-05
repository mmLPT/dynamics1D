import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import expm

from dynamics1D.quantum.grid import *
from dynamics1D.quantum.wavefunction import *


class Operator:
	# Abstract class for an x acting operator
	
	def __init__(self, grid,hermitian=False):
		self.grid=grid
		self.N=grid.N
		self.h=grid.h
		self.hermitian=hermitian

		self.M=[] # Matrix of the operator in X basis !

		self.eigenval=np.zeros(self.N,dtype=np.complex_) 
		self.eigenvec=[] # Eigenstates (from wave function class) id est can be used in both representation
	
	def fillM(self):
		# Fill the matrix representation of the operator
		self.M=np.zeros((self.N,self.N),dtype=np.complex_)
		pass
					
	def diagonalize(self):
		# Diagonalyze the matrix representation of the operator and 
		# save the wavefunctions
		self.fillM()
		eigenvec=np.zeros((self.N,self.N),dtype=np.complex_)
		if self.hermitian:
			self.eigenval,eigenvec=np.linalg.eigh(self.M)
		else:
			self.eigenval,eigenvec=np.linalg.eig(self.M)
			
		for i in range(0,self.N):
			wf=WaveFunction(self.grid)
			wf.x=eigenvec[:,i]
			wf.normalize("x")
			self.eigenvec.insert(i,wf)
			
class Hamiltonian(Operator):
	def __init__(self,grid,potential,beta=0.0):
		Operator.__init__(self,grid)
		self.beta=beta
		self.potential=potential
		self.Hp=(grid.p-self.beta*grid.h)**2/2
		self.Hx=self.potential.Vx(grid.x,0)
		self.hermitian=True
		
	def diagonalize(self):
		# Diagonalize, then compute quasi-energies
		Operator.diagonalize(self)
		ind=np.argsort(self.eigenval)
		self.eigenvec=[self.eigenvec[i] for i in ind]
		self.eigenval=self.eigenval[ind]	
	
	def fillM(self):
		self.M=np.zeros((self.N,self.N),dtype=np.complex_)
		for i in range(0,self.N):
			wfx=WaveFunction(self.grid)
			wfx.setState("diracx",i)
			wfx.x=wfx.x*self.Hx
			wfp=WaveFunction(self.grid)
			wfp.setState("diracx",i)
			wfp.p=wfp.p*self.Hp
			wfp.p2x() 
			wf=wfx+wfp
			self.M[:,i]=wf.x
	
class TimePropagator(Operator):
	# Class to be used to described time evolution operators such has
	# |psi(t')>=U(t',t)|psi(t)> with U(t',t)=U(dt,0)^idtmax
	# It relies on splliting method with H = p**2/2 + V(x,t)
	# It can be use for :
	# - periodic V(x,t) (including kicked systems)
	# - time-indepent V(x)

	def __init__(self,grid,potential,beta=0.0,T0=1,idtmax=1000):
		Operator.__init__(self,grid)
		self.hermitian=False
		
		self.potential=potential
		
		self.T0=T0 # Length of propagation
		self.idtmax=idtmax # number of step : 1 -> kicked/ =/=1 -> periodic or time independent
		self.dt=self.T0/self.idtmax # time step
		
		self.beta=beta # quasi-momentum
		self.Up=np.exp(-1j*((grid.p-self.beta*grid.h)**2/4)*self.dt/grid.h)
		x,t=np.meshgrid(grid.x,np.arange(self.idtmax)*self.dt)	
		self.Ux=np.exp(-1j*(self.potential.Vx(x,t))*self.dt/grid.h)

			
	def propagate(self,wf):
		# Propagate over one period/kick/arbitray time 
		for idt in range(0,self.idtmax):
			wf.p=wf.p*self.Up 
			wf.p2x() 
			wf.x=wf.x*self.Ux[idt]
			wf.x2p() 
			wf.p=wf.p*self.Up 

	def fillM(self):
		# Propagate N dirac in x representation, to get matrix representation
		# of the quantum time propagator
		self.M=np.zeros((self.N,self.N),dtype=np.complex_)

		for i in range(0,self.N):
			wf=WaveFunction(self.grid)
			wf.setState("diracx",i)
			self.propagate(wf)
			wf.p2x()
			self.M[:,i]=wf.x
			
class FloquetPropagator(TimePropagator):
	# This class is specific for CAT purpose:
	# WIP: a bit dirty.
	def __init__(self,grid,potential,beta=0.0,T0=1,idtmax=1000):
		TimePropagator.__init__(self,grid,potential,beta=beta,T0=T0,idtmax=idtmax)
		self.quasienergy=np.zeros(grid.N) # quasi energies
		
	def diagonalize(self):
		# Diagonalize, then compute quasi-energies
		Operator.diagonalize(self)
		for i in range(0,self.N):
			self.quasienergy[i]=-np.angle(self.eigenval[i])*(self.h/self.T0)	
			
	def propagateBetween(self,wf,t1,t2):
		# Propagate over one period/kick/arbitray time 
		i1=int(t1/self.dt)
		i2=int(t2/self.dt)
		for idt in range(i1,i2):
			wf.p=wf.p*self.Up 
			wf.p2x() 
			wf.x=wf.x*self.Ux[idt]
			wf.x2p() 
			wf.p=wf.p*self.Up  
		
	def orderEigenstatesWithOverlapOn(self,wf):
		overlaps=np.zeros(self.N,dtype=complex)
		for i in range(0,self.N):
			overlaps[i]=self.eigenvec[i]%wf
			
		ind=np.flipud(np.argsort(np.abs(overlaps)**2))
		
		self.eigenvec=[self.eigenvec[i] for i in ind]
		self.eigenval=self.eigenval[ind]
		self.quasienergy=self.quasienergy[ind]
		
		return overlaps[ind]	
	
		
	def diffquasienergy(self,qe1,qe2):
		# This returns the difference on a circle
		de=np.pi*(self.h/self.T0)
		diff=qe1-qe2
		if diff>de:
			return diff-2*de
		elif diff<-de:
			return diff+2*de
		else:
			return diff
				
	def getTunnelingPeriod(self):
		return 2*np.pi*self.h/(self.T0*(abs(self.diffquasienergy(self.quasienergy[0],self.quasienergy[1]))))
		
	def getSpacingDistribution(self,bins=50):
		ind1=np.argsort(self.qE)
		symX=np.zeros(self.N,dtype=bool)
		for i in range(0,self.N):
			symX[i]=self.eigenvec[ind1[i]].isSymetricInX()
			
		s=np.zeros([])
		for ind2 in [np.nonzero(symX)[0],np.nonzero(np.invert(symX))[0]]:
			print(ind1,ind2)
			for i in range(0,len(ind2)-1):
				# ~ a=np.abs(self.diffqE1qE2(ind[i],ind[i+1]))/(2*np.pi*self.h/(len(ind)*self.T0))
				a=np.abs(self.diffqE1qE2(ind1[ind2[i]],ind1[ind2[i+1]]))/(2*np.pi*self.h/(len(ind2)*self.T0))
				print(len(ind2),a)
				s=np.append(s,a)
			s=np.append(s,np.abs(self.diffqE1qE2(ind1[ind2[len(ind2)-1]],ind1[ind2[0]]))/(2*np.pi*self.h/(len(ind2)*self.T0)))
		return np.histogram(s, bins=bins,density=True)


class FloquetRandomPhasePropagator(FloquetPropagator):
	def __init__(self,grid,potential,beta=0.0,T0=1,idtmax=1):
		FloquetPropagator.__init__(self,grid,potential,beta=beta,T0=T0,idtmax=idtmax)		
		self.Up=np.exp(1j*(np.random.rand(self.N)*2*np.pi))
					
	def propagate(self,wf):
		# Propagate over one period/kick/arbitray time 
		for idt in range(0,self.idtmax):
			wf.p=wf.p*self.Up 
			wf.p2x() 
			wf.x=wf.x*self.Ux[idt]
			wf.x2p() 
			
	def getFormFactor(self,it):
		# Returns form factor for given time/n
		n=int(it)
		return abs(np.sum(self.eigenval**n))**2/self.N	
		
class DiscreteOperator:
	def __init__(self, lattice):
		self.lattice=lattice
		self.N=lattice.N
		self.h=lattice.h
		self.M=[]
		
class TightBindingHamiltonian(DiscreteOperator):
	def __init__(self,lattice,spectrumfile):
		DiscreteOperator.__init__(self,lattice)

		data=np.load(spectrumfile)
		self.quasienergies=data['quasienergies'][:,0]
		self.beta=data['beta'][:,0]*2*np.pi
		data.close()

		self.Vn=np.fft.rfft(self.quasienergies)/self.N
		self.Vnth=np.zeros(self.Vn.size,dtype=complex)
		
		self.M=np.zeros((self.N,self.N),dtype=np.complex_)
		self.M[0]=np.concatenate((self.Vn,np.conjugate(np.flip(np.delete(self.Vn,0),axis=0))))
		
		for i in range(1,self.N):
			self.M[i]=np.roll(self.M[0],i)	
			
		self.computeVnth()			
				
	def computeVnth(self):
	
		diff=self.quasienergies-np.roll(self.quasienergies,-1)
		dbeta=0.5*(self.beta[1]-self.beta[0])

		ind= ((np.diff(np.sign(np.diff(np.abs(diff)))) < 0).nonzero()[0]+1)
		beta0=self.beta[ind]
		W=diff[ind]
		ind=beta0>0
		beta0=beta0[ind]+dbeta
		W=W[ind]
		
		n=np.arange(self.Vn.size)
		n[0]=n[1]
		
		def Vnthfun(n,W,beta0):
			return -W/(np.pi*n)*np.sin(beta0*n)
		
		for i in range(0,W.size):
			self.Vnth+=Vnthfun(n,W[i],beta0[i])
		
		self.Vnth[0]=self.Vn[0]
				
class DiscreteTimePropagator(DiscreteOperator):
	def __init__(self,grid,T0,hamiltonian):
		DiscreteOperator.__init__(self,grid)
		self.T0=T0
		self.M=expm(-1j*hamiltonian.M*self.T0/self.h)
	
	def propagate(self,wf):
		wf.n=np.matmul(self.M,wf.n)

