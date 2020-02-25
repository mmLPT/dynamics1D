import matplotlib.pyplot as plt
import numpy as np
import random as rd


def modc(x,dX):
	x=x%dX
	return x*(x<0.5*dX)+(x-dX)*(x>0.5*dX)
					
class PhasePortrait:
	# This class can be used to generate a stroposcopic phase space of a 
	# given time periodic system. It requires time propagator, to include 
	# both discrete and continous time map.
	def __init__(self,nperiod,ny0,timepropagator,dX=2*np.pi,dP=4):
		self.nperiod=nperiod
		self.timepropagator=timepropagator
		self.ny0=ny0
		self.dX=dX
		self.dP=dP
			
	# ~ def generateInitialCondition(self,i):
		# ~ return np.array([rd.randint(0,101)/100.0*self.dX-self.dX*0.5,rd.randint(0,101)/100.0*self.dP-self.dP*0.5])
		
	def generateInitialCondition(self,i):
		x0=np.linspace(0,self.dX*0.5,self.ny0)[i]
		p0=0
		return np.array([x0,p0])
	
	def computeOrbit(self,i):
		y=self.generateInitialCondition(i)
		x,p=np.zeros(self.nperiod),np.zeros(self.nperiod)
		for it in range(0,self.nperiod):	
			x[it],p[it]=y[0],y[1]
			y=self.timepropagator.propagate(y)
		return x,p
	
	def getChaoticity(self,x,p):
		k=20
		H, xedges, yedges = np.histogram2d(x, p,bins=np.linspace(-np.pi,np.pi,k))
		s=0.0
		for i in range(0,k-1):
			for j in range(0,k-1):
				if H[i,j]>0.0:
					s+=1
		s/=(k-1)**2	
		return s
		
		
	def getOrbits(self):	
		chaoticity=np.zeros((self.ny0,self.nperiod))
		x=np.zeros((self.ny0,self.nperiod))
		p=np.zeros((self.ny0,self.nperiod))	
		for i in range(0,self.ny0):
			x[i],p[i] = self.computeOrbit(i)
			chaoticity[i]=np.ones(self.nperiod)*self.getChaoticity(x[i],p[i])
			
		chaoticity=chaoticity/np.max(chaoticity)
		return x,p,chaoticity
			
	def png2husimi(self,dfile="SPP"):
		x,p,chaoticity=self.getOrbits()
		
		fig, ax = plt.subplots(figsize=(self.dX,self.dP),frameon=False)
		ax.set_xlim(-self.dX/2.0,self.dX/2.0)
		ax.set_ylim(-self.dP/2.0,self.dP/2.0)
		ax.scatter(modc(x,2*np.pi),p,c="black",s=1**2)
		fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
		plt.savefig(dfile+".png",dpi=500)

