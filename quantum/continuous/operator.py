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
		self.M=np.zeros((self.N,self.N),dtype=np.complex_) # The operator representation
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
				wf.normalizeX()
				wf.x2p()
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
		
		###### NOTE DEVELOPPEMENT le quasi-moment, c'est plus une propriete de la grille ?
		self.beta=beta # quasi-momentum
		
		self.Up=np.exp(-1j*((grid.p-self.beta)**2/4)*self.dt/grid.h)
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

		for i in range(0,self.N):
			wf=WaveFunction(self.grid)
			wf.setState("diracx",i0=i,norm=False) #Norm false to produce 'normalized' eigevalue
			# A CLARIFIER NORME
			self.propagate(wf)
			wf.p2x()
			self.M[:,i]=wf.x
			
class FloquetPropagator(TimePropagator):
	# This class is specific for CAT purpose:
	# WIP: a bit dirty.
	def __init__(self,grid,potential,beta=0.0,T0=1,idtmax=1000):
		TimePropagator.__init__(self,grid,potential,beta=beta,T0=T0,idtmax=idtmax)
		self.qE=np.zeros(grid.N) # quasi energies
		
	def diagonalize(self):
		# Diagonalize, then compute quasi-energies
		Operator.diagonalize(self)
		for i in range(0,self.N):
			self.qE[i]=-np.angle(self.eigenval[i])*(self.h/self.T0)	
			
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
	
	# ~ def getOrderedOverlapsWith(self,wf):
		# ~ # Check overlaps with a given wave function
		# ~ # Returns the index of ordered overlaps and the overlaps
		# ~ overlaps=np.zeros(self.N,dtype=complex)
		# ~ for i in range(0,self.N):
			# ~ overlaps[i]=self.eigenvec[i]%wf
		# ~ ind=np.flipud(np.argsort(np.abs(overlaps)**2))
		# ~ return ind, overlaps[ind]
		
	def orderEigenstatesWithOverlapOn(self,wf):
		overlaps=np.zeros(self.N,dtype=complex)
		for i in range(0,self.N):
			overlaps[i]=self.eigenvec[i]%wf
			
		ind=np.flipud(np.argsort(np.abs(overlaps)**2))
		
		self.eigenvec=[self.eigenvec[i] for i in ind]
		self.eigenval=self.eigenval[ind]
		self.qE=self.qE[ind]
		
		return overlaps[ind]	
	
		
	def diffqE1qE2(self,i1,i2):
		# This returns the difference on a circle
		qE1=self.qE[i1]
		qE2=self.qE[i2]
		dE=np.pi*(self.h/self.T0)
		diff=qE1-qE2
		if diff>dE:
			return diff-2*dE
		elif diff<-dE:
			return diff+2*dE
		else:
			return diff
		
	def getSpacingDistribution2(self,bins=50):
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



	# ~ def propagateGP(self,wf,g):
		# ~ # Propagate over one period/kick/arbitray time with interactions
		# ~ for idt in range(0,self.idtmax):
			# ~ wf.p=wf.p*self.Up 
			# ~ wf.p2x() 
			# ~ wf.x=wf.x*self.Ux[idt]*np.exp(-(1j/self.grid.h)*(g*np.abs(wf.x)**2)*self.dt)
			# ~ wf.x2p() 
			# ~ wf.p=wf.p*self.Up  

	
		
	# ~ def getBallisticSpeed(self,Ncell,x0):
		# ~ wf0=WaveFunction(grid)
		# ~ wf0.setState("coherent",x0=x0,xratio=2.0)
		# ~ v=0.0
		# ~ for i in range(1,int(0.5*(Ncell-1))):
			# ~ xi=x0+i*2*np.pi
			# ~ wfi=WaveFunction(grid)
			# ~ wfi.setState("coherent",x0=xi,xratio=2.0)
			# ~ v=v+i**2*np.abs(self.getxbraket(wf0,wfi))**2
		# ~ return np.sqrt(v)
		
		
	# ~ def getSpacingDistribution(self):
		# ~ ind=np.argsort(self.qE)
		# ~ # On suppose Nh=2pi et T=2pi do we
		# ~ s=np.zeros(self.N)
		# ~ for i in range(0,self.N-1):
			# ~ s[i]=np.abs(self.diffqE1qE2(ind[i],ind[i+1]))/(2*np.pi*self.h/(self.N*self.T0))
		# ~ s[self.N-1]=np.abs(self.diffqE1qE2(ind[self.N-1],ind[0]))/(2*np.pi*self.h/(self.N*self.T0))
		# ~ return s
		
		

		
		
	# ~ def getQETh(self,i0,pot):
		# ~ # Returns expected value of quasi-energies according to 
		# ~ # perturbation theory up to 3rd order for a given potential
		# ~ V00=pot.braketVxasym(self.eigenvec[0],self.eigenvec[0])
		# ~ V11=pot.braketVxasym(self.eigenvec[1],self.eigenvec[1])
		# ~ V01=pot.braketVxasym(self.eigenvec[0],self.eigenvec[1])
		# ~ V10=pot.braketVxasym(self.eigenvec[1],self.eigenvec[0])
		# ~ E0mE1=self.diffqE1qE2(0,1)
		# ~ E1mE0=-E0mE1
		
		# ~ if i0==0:
			# ~ e0=self.qE[self.iqgs]
			# ~ e1=abs(V00)
			# ~ e2=abs(V01)**2/E0mE1
			# ~ e3=abs(V01)**2/(E0mE1)**2*(abs(V11)-abs(V00))
			# ~ #e4=abs(V01)**2*abs(V11)**2/E0mE1**3-e2*abs(V10)**2/E0mE1**4-2*abs(V00)*abs(V01)**2*abs(V11)/E0mE1**3+abs(V00)**2*abs(V01)**2/E0mE1**3
		# ~ elif i0==1:
			# ~ e0=self.qE[self.iqfes]
			# ~ e1=abs(V11)
			# ~ e2=abs(V01)**2/E1mE0
			# ~ e3=abs(V01)**2/(E0mE1)**2*(abs(V00)-abs(V11))
			# ~ #e4=abs(V11)**2*abs(V00)**2/E1mE0**3-e2*abs(V10)**2/E1mE0**4-2*abs(V11)*abs(V01)**2*abs(V00)/E1mE0**3+abs(V11)**2*abs(V01)**2/E1mE0**3
		
		# ~ e=e0 +e1 +e2 #+e3
		# ~ return e	
		
		
			
	# ~ def getTunnelingPeriodBetween(self,i1,i2):		
		# ~ return 2*np.pi*self.h/(self.T0*(abs(self.diffqE1qE2(i1,i2))))
		
	# ~ def getTunnelingFrequencyBetween(self,i1,i2):		
		# ~ return np.abs(self.diffqE1qE2(i1,i2))/self.h
		
		# ~ def propagatequater(self,wf):
		# ~ # Propagate over one period/kick/arbitray time 
		# ~ for idt in range(0,int(self.idtmax/4)):
			# ~ wf.p=wf.p*self.Up 
			# ~ wf.p2x() 
			# ~ wf.x=wf.x*self.Ux[idt]
			# ~ wf.x2p() 
			# ~ wf.p=wf.p*self.Up  
			
	# ~ def propagatequater2(self,wf):
		# ~ # Propagate over one period/kick/arbitray time 
		# ~ for idt in range(int(self.idtmax/4)-1,self.idtmax):
			# ~ wf.p=wf.p*self.Up 
			# ~ wf.p2x() 
			# ~ wf.x=wf.x*self.Ux[idt]
			# ~ wf.x2p() 
			# ~ wf.p=wf.p*self.Up
			
			
		# ~ if self.randomphase:
			# ~ for i in range(0,self.N):
				# ~ wf=WaveFunction(self.grid)
				# ~ wf.setState("diracx",i0=i,norm=False) #Norm false to produce 'normalized' eigevalue
				# ~ self.propagateRandom(wf)
				# ~ wf.p2x()
				# ~ self.M[:,i]=wf.x 
				
				
		## //!\\ A tester 24/02/2020		
		# ~ elif self.Mrepresentation=="p": 
			# ~ for i in range(0,self.N):
				# ~ wf=WaveFunction(self.grid)
				# ~ ## //!\\ -> shift pas necessaire
				# ~ wf.p=np.fft.ifftshift(eigenvec[:,i])*self.grid.phaseshift 
				# ~ wf.p2x()
				# ~ wf.normalizeX()
				# ~ wf.x2p()
				# ~ self.eigenvec.insert(i,wf)


