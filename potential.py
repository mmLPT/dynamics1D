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
	def __init__(self,e,gamma,mw=np.cos,Fb=0):
		Potential.__init__(self)
		
		self.e=e
		self.gamma=gamma
		self.mw=mw
		self.Fb=Fb	
		
	def Vx(self,x,t=0):
		return -self.gamma*(1+self.e*self.mw(t))*np.cos(x)+self.Fb*x/(2*np.pi)
	
	def dVdx(self,x,t=0):
		return self.gamma*(1+self.e*self.mw(t))*np.sin(x)+self.Fb/(2*np.pi)
		
		
class DoubleWell(Potential):
	def __init__(self,gamma,x0):
		Potential.__init__(self)
		self.x0=x0
		self.gamma=gamma
		
	def Vx(self,x,t=0):
		return self.gamma*(x**2-self.x0**2)**2
	
	def dVdx(self,x,t=0):
		return self.gamma*4*x*(x**2-self.x0**2)
		
class SawTooth(Potential):
	def __init__(self,gamma,dx=2*np.pi):
		Potential.__init__(self)
		self.gamma=gamma
		self.dx=dx
		
	def Vx(self,x,t=0):
		return -4*np.pi**2*self.gamma*np.mod(x/self.dx,1.0)
		
class KickedRotor(Potential):
	def __init__(self,K):
		Potential.__init__(self)
		self.K=K
		
	def Vx(self,x,t=0):
		return -self.K*np.cos(x)
	
	def dVdx(self,x,t=0):
		return self.K*np.sin(x)
		
class Square(Potential):
	def __init__(self,x0,V0):
		Potential.__init__(self)
		self.x0=x0
		self.V0=V0
		
	def Vx(self,x,t=0):
		return -2*np.pi*self.V0*np.array(x>-self.x0)*np.array(x<self.x0)
		
		

