import numpy as np
from numpy.linalg import matrix_power

from dynamics1D.quantum.continuous.grid import *
from dynamics1D.quantum.continuous.wavefunction import *

class Operator:
	# This class provides a discrete representation for an x/p acting operator
	def __init__(self, grid,hermitian=False,Mrepresentation="x"):
		self.grid=grid
		self.N=grid.N
		self.h=grid.h
		self.hermitian=hermitian
		self.Mrepresentation=Mrepresentation #x or p depending on how you fillup M
		self.M=np.empty((self.N,self.N),dtype=np.complex_) # The operator representation

		self.eigenval=np.zeros(self.N,dtype=np.complex_) 
		self.eigenvec=[] # Eigenstates (from wave function class) id est can be used in both representation
	
	def fillM(self):
		# Fill the matrix representation of the operator
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
			
		if self.Mrepresentation=="x": 
			for i in range(0,self.N):
				wf=WaveFunction(self.grid)
				wf.x=eigenvec[:,i]
				wf.normalize("x")
				self.eigenvec.insert(i,wf)
				
		def getxbraket(self,wf1,wf2):
			if self.Mrepresentation=="x": 
				return np.matmul(np.conjugate(wf1.x),np.matmul(self.M,wf2.x))
			elif self.Mrepresentation=="p": 
				return np.matmul(np.conjugate(wf1.x),np.matmul(self.M,wf2.x))
				
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
		self.Mrepresentation="x"
		
		print(self.M.shape)

		for i in range(0,self.N):
			wf=WaveFunction(self.grid)
			# ~ wf.setState("diracx",i0=i,norm=False) #Norm false to produce 'normalized' eigevalue
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

		# The random phases ---> a verifier avec les chefs, phases aléatoire à chaque kick ?
		self.Up=np.exp(1j*(np.random.rand(self.N)*2*np.pi))
					
	def propagate(self,wf):
		# Propagate over one period/kick/arbitray time 
		for idt in range(0,self.idtmax):
			wf.p=wf.p*self.Up 
			wf.p2x() 
			wf.x=wf.x*self.Ux[idt]
			wf.x2p() 
			
	def getFormFactor(self,it):
		n=int(it)
		return abs(np.sum(self.eigenval**n))**2/self.N	

