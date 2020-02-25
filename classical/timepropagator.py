import numpy as np

# This script contains : 2 classes
# + class : ClassicalDiscreteTimePropagator
# + class : ClassicalContinueTimePropagator

def RK4(f,y,t,dt):
	k1=f(y,t)
	k2=f(y+dt/2.0*k1,t+dt/2.0)
	k3=f(y+dt/2.0*k2,t+dt/2.0)
	k4=f(y+dt*k3,t+dt)
	return y+dt/6.0*(k1+2*k2+2*k3+k4)

class DiscreteTimePropagator:
	def __init__(self,potential):
		self.potential=potential

	def propagate(self,y):
		ydot=np.zeros(y.shape) #y'
		ydot[1]=y[1]-self.potential.dVdx(y[0]+0.5*y[1],t)
		ydot[0]=y[0]+0.5*(yp[1]+y[1])
		return ydot
		
	# ~ def propagate(self,y):		# non symétrisé
		# ~ yp[1]=y[1]-self.potential.dVdx(y[0])
		# ~ yp[0]=y[0]+yp[1]
		# ~ return yp
	
class ContinuousTimePropagator:
	def __init__(self,potential,T0=2*np.pi,ndt=100):
		self.ndt=ndt
		self.dt=T0/ndt
		self.potential=potential
		
	def f(self,y,t):
		# scheme to solve motion equation with RK4 such as y'=f(y,t) with y at 2D
		ydot=np.zeros(y.shape) 
		ydot[0]=y[1]
		ydot[1]=-self.potential.dVdx(y[0],t)
		return ydot

	def propagate(self,y):		
		for i in range(0,self.ndt):
			y=RK4(self.f,y,i*self.dt,self.dt) #propagation
		return y

