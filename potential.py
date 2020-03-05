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
	def __init__(self,e,gamma,mw=np.cos,Fb=0,phi=0):
		Potential.__init__(self)
		
		self.e=e
		self.gamma=gamma
		self.mw=mw
		self.Fb=Fb	
		self.phi=phi
		
	def Vx(self,x,t=0):
		return -self.gamma*(1+self.e*self.mw(t))*np.cos(x+self.phi*np.sin(t))+self.Fb*x/(2*np.pi)
	
	def dVdx(self,x,t=0):
		return self.gamma*(1+self.e*self.mw(t))*np.sin(x+self.phi*np.sin(t))+self.Fb/(2*np.pi)
		
class ModulatedPendulumHarmonicConfinment(Potential):
	def __init__(self,e,gamma,mu):
		Potential.__init__(self)
		
		self.e=e
		self.gamma=gamma
		self.mu=mu	
	
		
	def Vx(self,x,t=0):
		return -self.gamma*(1+self.e*np.cos(t))*np.cos(x)+self.mu*(x/(2*np.pi))**2
	
	# ~ def dVdx(self,x,t=0):
		# ~ return self.gamma*(1+self.e*self.mw(t))*np.sin(x+self.phi*np.sin(t))+self.Fb/(2*np.pi)
		
		
class DoubleWell(Potential):
	def __init__(self,e,gamma):
		Potential.__init__(self)
		self.gamma=gamma
		self.e=e
		
	def Vx(self,x,t=0):
		return -self.gamma*(np.cos(x)-self.e*np.cos(2*x))
	
	def dVdx(self,x,t=0):
		return self.gamma*(np.sin(x)-self.e*2*np.sin(2*x))
		
class SawTooth(Potential):
	# Potential en dent de scie
	def __init__(self,gamma,dx=2*np.pi):
		Potential.__init__(self)
		self.gamma=gamma
		self.dx=dx
		
	def Vx(self,x,t=0):
		return -2*np.pi*self.gamma*np.mod(x/self.dx,1.0)
			
class Rectangle(Potential):
	# Potential en créneau
	def __init__(self,gamma,x1=-np.pi/2,x2=np.pi/2):
		Potential.__init__(self)
		self.x1=x1
		self.x2=x2
		self.gamma=gamma
		
	def Vx(self,x,t=0):
		return -2*np.pi*self.gamma*np.array(x>self.x1)*np.array(x<self.x2)
		
		
class KickedRotor(Potential):
	def __init__(self,K):
		Potential.__init__(self)
		self.K=K
		
	def Vx(self,x,t=0):
		return -self.K*np.cos(x)
	
	def dVdx(self,x,t=0):
		return self.K*np.sin(x)
		
class HarmonicOscillator(Potential):
	def __init__(self,omega):
		Potential.__init__(self)
		self.omega=omega
		
	def Vx(self,x,t=0):
		return 0.5*self.omega**2*x**2
		
		

