# 
# Generate Artificial Rain Field with Gaussian Process
# 
import os
import sys
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import GPy
import h5py
from scipy.interpolate import griddata

def simulate_initial_gp_field(scale,grid_size,nsamples,volume,seed):
    """
    Generate 2d random field
    scale : spatial scale of disturbance
    grid_size : grid size
    volume : overall scale of rain field
    nsamples : number of fields generated
    seed : random seed
    """
    scale = 0.2
    # define GP kernel
    kernel = GPy.kern.RBF(input_dim=2, variance=1, lengthscale=scale)

    np.random.seed(seed=seed)

    # Define Grids
    x_sim = np.linspace(-1, 1, grid_size)
    y_sim = np.linspace(-1, 1, grid_size)
    # Expand Grid
    xy = np.array(np.meshgrid(x_sim,y_sim)).reshape(2,-1).T

    mu = np.zeros(grid_size*grid_size)
    cov = kernel.K(xy, xy)

    # generate random variable
    sim = np.random.multivariate_normal(mu, cov, size=nsamples)
    sim = sim.reshape([nsamples,grid_size,grid_size])

    # convert to grid
    xy_grd = xy.reshape([grid_size,grid_size,2])
    # scale to "cell center" grid position
    xy_grd = (xy_grd + 1.0)*(grid_size-1.0)/2.0 + 0.5

    # clip the value in [0-1] range
    sim = sim * 0.5 * volume
    sim=sim.clip(min=0.0,max=1.0)
    
    return xy_grd,sim

def interp_finer_grid(xy_grd,field,grid_size):
    """
    Interpolate to finer grid
    xy_grd : grid with [x,y,2] dimension
    field : value with [x,y] dimension
    grid_size : grid size of finer grid
    """
    xsize,ysize=field.shape
    # Define Grids
    x_f = np.linspace(np.min(xy_grd[:,:,0]),np.max(xy_grd[:,:,0]), grid_size)
    y_f = np.linspace(np.min(xy_grd[:,:,1]),np.max(xy_grd[:,:,1]), grid_size)
    # Expand Grid
    xy_tgt = np.array(np.meshgrid(x_f,y_f)).transpose((1,2,0))
    # interpolation
    out = griddata(xy_grd.reshape([xsize*ysize,2]),
                   field.reshape([xsize*ysize]),
                   xy_tgt.reshape([grid_size*grid_size,2]), method='cubic')
    out_field = out.reshape([grid_size,grid_size])
    xy_out = (xy_tgt - 0.5)/(xsize-1)*(grid_size-1) + 0.5
    return xy_out,out_field

def velocity_field_uniform(v,grid_size):
    """
    Generate Velocity Field
    v : velocity (vx,vy)
    grid_size : grid size of finer grid
    """
    vfield = np.zeros((2,grid_size,grid_size))
    vfield[0,:,:] = v[0]
    vfield[1,:,:] = v[1]
    return vfield

def advection(xy_grd,field,v,dt,steps):
    """
    Advection by semi-Lagrange scheme
    xy_grd : grid with [x,y,2] dimension
    field : value with [x,y] dimension
    v : velocity field with [x,y,2] dimension
    dt : time step
    steps : number of time steps
    """
    xsize,ysize=field.shape
    out_field = np.zeros((steps,xsize,ysize))
    for n in range(steps):
    #for n in [1]:
        xy_tgt = xy_grd.copy()
        # calc x and y component separately
        xy_tgt[:,:,0] = xy_tgt[:,:,0] - v[0]*dt*n
        xy_tgt[:,:,1] = xy_tgt[:,:,1] - v[1]*dt*n
        # interpolation
        out = griddata(xy_grd.reshape([xsize*ysize,2]),
                       field.reshape([xsize*ysize]),
                       xy_tgt.reshape([xsize*ysize,2]), method='nearest')
        out_field[n,:,:] = out.reshape([xsize,ysize])
    return out_field

def plot_field(out_field,pic_path,case):
    """
    Post plotting
    out_field : field with [time,x,y] dimension
    """
    # create pic save dir
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)

    steps,xsize,ysize=out_field.shape
    for n in range(steps):
        title="%s step:%02d" % (case,n)
        vmin=0
        vmax=1
        # plot
        plt.imshow(out_field[n,:,:],vmin=vmin,vmax=vmax,cmap="GnBu",origin='lower')
        plt.colorbar()
        plt.grid()
        plt.title(title)
        #plt.show()
        nt_str = '_dt%02d' % n
        plt.savefig(pic_path+'art_field_'+case+nt_str+'.png')
        plt.cla() 
        plt.clf()
        plt.close("all")
        gc.collect()


def generate_rain_field(grid_size,v,scale,volume,num,out_dir):
    # initial values
    #scale = 0.1
    grid_size_gp = 50
    nsamples = 1
    xy_grd,sim=simulate_initial_gp_field(scale,grid_size_gp,nsamples,volume,seed=num)

    # advection
    v = v
    dt = 1.0
    steps = 24
    for ns in range(nsamples):
        # interpolate to finer grid
        xy_grd_f,field_f = interp_finer_grid(xy_grd,sim[ns,:,:],grid_size)
        # velocity field
        vfield = velocity_field_uniform(v,grid_size)
        # apply advection
        out_field = advection(xy_grd_f,field_f,vfield,dt,steps)
        # plotting
        case = "V%5.2f_%5.2f_S%5.3f_A%5.3f" % (v[0],v[1],scale,volume)
        print("case=",case)
        pic_path= out_dir + "%s/" % case
        plot_field(out_field,pic_path,case)
        
    # type
    out_field = out_field.astype(np.float32)
    vfield = vfield.astype(np.float32)
    # write
    h5fname = "art_field_%04d.h5" % num
    print('writing h5 file:',h5fname)
    h5file = h5py.File(out_dir+h5fname,'w')
    h5file.create_dataset('R',data= out_field)
    h5file.create_dataset('V',data= vfield)
    h5file.close()  
    
if __name__ == '__main__':
    # read case name from command line
    argvs = sys.argv
    argc = len(argvs)

    # max motion vector [pixels/frame]
    vmax = 0.0
    out_dir = "../data/artfield/vzero_256/"
    grid_size = 256

    if argc != 4:
        print('Usage: python artificial_rain_field.py seed start end')
        quit()

    seed = int(argvs[1])
    istart = int(argvs[2])
    iend = int(argvs[3])
    print("random seed:",seed)
    print("start and end of the dataset:",istart,iend)

    # create result dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    # run
    np.random.seed(seed=seed)
    for num in range(istart,iend):
        vx = np.random.uniform(-vmax,vmax)
        vy = np.random.uniform(-vmax,vmax)
        velocity = (vx, vy)
        scale = np.random.uniform(0.01,0.1)
        volume = np.random.uniform(0.1,1.0)
        generate_rain_field(grid_size,velocity,scale,volume,num,out_dir)
