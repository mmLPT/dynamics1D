import numpy as np

class Potential:
	# The class potential is desgined to be herited only
	def __init__(self):	
		pass
	def Vx(self,x,t=0):
		pass
	def dVdx(self,x,t=0):
		pass
		

class ModulatedPendulum(Potential):
	def __init__(self,e,gamma,f=np.cos,Fb=0):
		Potential.__init__(self)
		
		self.e=e
		self.gamma=gamma
		self.f=f # modulation waveform
		self.Fb=Fb
		
		# ~ self.a1=getFourierCoefficient("a",1,self.f)
		# ~ self.b1=getFourierCoefficient("b",1,self.f)
		
		self.a1=1
		self.b1=1
		
		
		if self.e==0:
			self.d1=0
		else:
			self.d1=(self.gamma-0.25)/(self.e*self.gamma)
		self.x0=self.R1()
		
	def Vx(self,x,t=np.pi/2.0):
		return -self.gamma*(1+self.e*self.f(t))*np.cos(x)+self.Fb*x/(2*np.pi)
	
	def dVdx(self,x,t=np.pi/2.0):
		return self.gamma*(1+self.e*self.f(t))*np.sin(x)
		
	# The 4 following functions comes from classical analysis fo the bifurcation
	# they make possible to acess equilibrium positions for a given modulation waveform
	def thetaR1(self):
		if self.a1==0:
			v=0.25*np.pi*self.b1/abs(self.b1)
		elif self.a1>0.0:
			v=0.5*np.arctan(self.b1/self.a1)
		else:
			v=0.5*np.arctan(self.b1/self.a1)+0.5*np.pi
		return np.arctan2(np.sin(v)/2,np.cos(v))
		
	def thetaR2(self):
		if self.a1==0:
			v=0.25*np.pi*self.b1/abs(self.b1)
		elif self.a1>0.0:
			v=0.5*np.arctan(self.b1/self.a1)
		else:
			v=0.5*np.arctan(self.b1/self.a1)+0.5*np.pi
		return np.arctan2(np.sin(v+np.pi/2)/2,np.cos(v+np.pi/2))
		
	def R1(self):
		if self.d1>-0.5*np.sqrt(self.a1**2+self.b1**2):
			v=8.0/self.gamma*self.e*(0.5*np.sqrt(self.a1**2+self.b1**2)+self.d1)*self.gamma
		else:
			v=0.0
		return np.sqrt(v*(np.cos(self.thetaR1())**2+np.sin(self.thetaR1())**2/4))
		
	def R2(self):
		if self.d1>0.5*np.sqrt(self.a1**2+self.b1**2):
			v=8.0/self.gamma*self.e*(-0.5*np.sqrt(self.a1**2+self.b1**2)+self.d1)*self.gamma
		else:
			v=0.0
		return np.sqrt(v*(np.cos(self.thetaR2())**2+np.sin(self.thetaR2())**2/4))
		
		
class PotentialDW(Potential):
	def __init__(self,gamma,idtmax=1000):
		Potential.__init__(self)
		self.T0=1
		self.idtmax=idtmax
		self.x0=np.pi/2
		self.gamma=gamma
		
		
	def Vx(self,x,t=np.pi/2.0):
		return -self.gamma*(np.cos(x)-np.cos(2*x))
	
	def dVdx(self,x,t=np.pi/2.0):
		return self.gamma*(np.sin(x)-2*np.sin(2*x))
		
class PotentialST(Potential):
	def __init__(self,gamma):
		Potential.__init__(self)
		self.gamma=gamma
		self.dx=2*np.pi
		self.T0=1.0
		self.idtmax=1
		
	def Vx(self,x):
		return -4*np.pi**2*self.gamma*np.mod(x/self.dx,1.0)
		
	# ~ def Vx(self,x):
		# ~ return -4*np.pi**2*self.gamma*(np.cos(x)*np.array(x>0)-np.cos(x)*np.array(x<0))
		
class PotentialKR(Potential):
	def __init__(self,K):
		Potential.__init__(self)
		self.K=K
		self.T0=1.0
		self.idtmax=1
		
	def Vx(self,x):
		return self.K*np.cos(x)
		
class PotentialGG(Potential):
	def __init__(self,a,V0):
		Potential.__init__(self)
		self.a=a
		self.V0=V0
		self.T0=1.0
		self.idtmax=1
		
	def Vx(self,x):
		return -2*np.pi*self.V0*np.array(x>-self.a)*np.array(x<self.a)
		

