import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
# ~ from matplotlib.colors import BoundaryNorm
# ~ from matplotlib.ticker import MaxNLocator

# ~ from matplotlib import gridspec
# ~ import matplotlib.image as mpimg

# This scripts contains: 1 class
# + class : Husimi

class Husimi:
	# The Husimi class provides a tool to generate Husimi representation
	# of wavefunctions. It is build from a grid, so you can generate 
	# representation of differents wave functions from a single object
	
	def __init__(self, grid, scale=1,pmax=2*np.pi):
		self.grid=grid
		self.scale=scale 
		# Husimi grid is defined over coherent states, but you can  
		# fix an higher resolution by changing 'scale'
		self.h=grid.h
		self.N=grid.N
		
		# Sigma of a coherent state with 1:1 aspect ratio
		self.sigmap=np.sqrt(self.h/2.0)
		self.sigmax=np.sqrt(self.h/2.0)
		
		# Boundaries for plotting
		self.xmax=grid.xmax
		self.pmax=min(np.max(grid.p)-np.min(grid.p),pmax)
		
		# Building Husimi grid
		self.Nx=int(self.xmax/self.sigmax*self.scale)
		self.x=np.linspace(-self.xmax/2.0,self.xmax/2.0,self.Nx)
		self.Np=int(self.pmax/self.sigmap*self.scale)
		self.p=np.linspace(-self.pmax/2.0,self.pmax/2.0,self.Np)
		
		self.dx=self.x[0]-self.x[1]
		self.dp=self.p[0]-self.p[1]
		
		
		self.husimi=np.zeros((self.Np,self.Nx))
		
		# The following is a trick : as we are working on a periodic 
		# grid, we want
		self.pshift=np.zeros((self.Np,self.N),dtype=np.complex_)
		pgrid=np.fft.fftshift(grid.p)
		for ip in range(0,self.Np):
			i0=int((self.p[ip]+self.N*self.h/2.0)/self.h)
			for i in range(0,self.N):
				self.pshift[ip][i]=pgrid[i]
				if i< i0-self.N/2:
					self.pshift[ip][i]=pgrid[i]+self.N*self.h
				if i> i0+self.N/2:
					self.pshift[ip][i]=pgrid[i]-self.N*self.h	

	def compute(self,wf,datafile=""):
		# Computes Husimi representation of a given wavefunction
		# It returns a 1-normalized 2D-density
			
		psip=np.fft.fftshift(wf.p)
		for ip in range(0,self.Np):	
			p0=self.p[ip]
			phi1=np.exp(-(self.pshift[ip]-p0)**2/(2*self.sigmap**2))
			for ix in range(0,self.Nx):
				phi=phi1*np.exp(-(1j/self.h)*(self.x[ix]+self.xmax/2.0)*self.pshift[ip])
				self.husimi[ip][ix]= abs(sum(np.conj(phi)*psip))**2
		self.husimi=self.husimi/np.max(self.husimi)
		
		if datafile!="":
			np.savez(datafile,"w",x=self.x,p=self.p,husimi=self.husimi)
	
		
	def quickplot(self,datafile='',SPSfile='',Dx=2*np.pi,Dp=4):
		if datafile!="":
			data=np.load(datafile+".npz")
			husimi=data['husimi']
			x=data['x']
			p=data['p']
			data.close()
		else:
			x=self.x
			p=self.p
			husimi=self.husimi
		
		ax=plt.gca()
		ax.set_aspect('equal')
		ax.set_ylim(-self.pmax/2,self.pmax/2)
		ax.set_xlim(-self.xmax/2,self.xmax/2)
		
		cmap = plt.get_cmap("jet")
		
		if SPSfile!="":
			img=mpl.image.imread(SPSfile+".png")
			ax.imshow(img,extent=[-Dx/2,Dx/2,-Dp/2, Dp/2])
			
		ax.imshow(np.flipud(husimi),cmap=cmap,alpha=0.7,extent=[-self.xmax/2,self.xmax/2,-self.pmax/2, self.pmax/2])
		# ~ contours=plt.contour(x,p,husimi,colors="k", levels=np.linspace(0.0,1.0,7,endpoint=True),linestyles="--",linewidths=1.0)

		plt.show()
	
