from torch import nn
import torch

import numpy as np

def grid_to_pc(R_grd,XY_pc):
    # convert grid to pc
    # R_grd: grid value with [batch,channels,height,width] dim
    # XY_pc: point cloud position with [batch,2,N] dim
    #        scaled to [0,1]
    B, _, N = XY_pc.size()
    _, C, _, _ = R_grd.size()
    input = R_grd
    L = int(np.sqrt(N)) # note that sqrt(N) should be an integer
    vgrid = XY_pc.permute(0, 2, 1).reshape(B, L, L, 2).cuda()
    # rescale grid to [-1,1]
    vgrid = (vgrid - 0.5) * 2.0
    R_pc = torch.nn.functional.grid_sample(input, vgrid)
    R_pc = R_pc.reshape(B, C, N)
    return R_pc

def grid_to_pc_nearest(R_grd,XY_pc,XY_grd,interp_type,interpolator):
    # convert grid to pc
    # R_grd: grid value with [batch,channels,height,width] dim
    # XY_pc: point cloud position with [batch,2,N] dim
    #        scaled to [0,1]
    batch,k,height,width = R_grd.shape
    XY_grd_tmp = XY_grd.reshape(batch,2,height*width).permute(0,2,1).detach()
    XY_pc_tmp = XY_pc.permute(0,2,1).detach()
    R_grd_tmp = R_grd.reshape(batch,k,height*width)
    if interp_type == "nearest_normal":
        #R_pc = nearest_neighbor_interp(XY_pc_tmp,XY_grd_tmp,R_grd_tmp)
        R_pc = interpolator(XY_pc_tmp,XY_grd_tmp,R_grd_tmp)
    if interp_type == "nearest_kdtree":
        #R_pc = nearest_neighbor_interp_kd(XY_pc_tmp,XY_grd_tmp,R_grd_tmp)
        R_pc = interpolator(XY_pc_tmp,XY_grd_tmp,R_grd_tmp)
    return R_pc

def grid_to_pc_nearest_id(R_grd,XY_pc,XY_grd,index,interpolator_fwd):
    # convert grid to pc
    # R_grd: grid value with [batch,channels,height,width] dim
    # XY_pc: point cloud position with [batch,2,N] dim
    #        scaled to [0,1]
    batch,k,height,width = R_grd.shape
    XY_grd_tmp = XY_grd.reshape(batch,2,height*width).permute(0,2,1).detach()
    XY_pc_tmp = XY_pc.permute(0,2,1).detach()
    R_grd_tmp = R_grd.reshape(batch,k,height*width)
    #R_pc = nearest_neighbor_interp_fi(XY_pc_tmp,XY_grd_tmp,R_grd_tmp,index,"forward")
    R_pc = interpolator_fwd(XY_pc_tmp,XY_grd_tmp,R_grd_tmp,index,"forward")
    return R_pc

def pc_to_grid(R_pc,XY_pc,height):
    # convert pc to grid
    # R_pc: point cloud value with [batch,channels,N] dim
    # XY_pc: point cloud position with [batch,2,N] dim
    #        scaled to [0,1]
    
    # apply interpolation
    R_grd = RevBilinear.apply(XY_pc, R_pc, height)
    R_grd = R_grd.permute(0,1,3,2)
    return R_grd

def pc_to_grid_nearest(R_pc,XY_pc,XY_grd,interp_type,interpolator):
    # convert pc to grid
    # R_pc: point cloud value with [batch,channels,N] dim
    # XY_pc: point cloud position with [batch,2,N] dim
    #        scaled to [0,1]
    batch,_,height,width = XY_grd.shape
    _,k,_ = R_pc.shape
    XY_grd_tmp = XY_grd.reshape(batch,2,height*width).permute(0,2,1).detach()
    XY_pc_tmp = XY_pc.permute(0,2,1).detach()
    if interp_type == "nearest_normal":
        #R_grd = nearest_neighbor_interp(XY_grd_tmp,XY_pc_tmp,R_pc)
        R_grd = interpolator(XY_grd_tmp,XY_pc_tmp,R_pc)
    if interp_type == "nearest_kdtree":
        #R_grd = nearest_neighbor_interp_kd(XY_grd_tmp,XY_pc_tmp,R_pc)
        R_grd = interpolator(XY_grd_tmp,XY_pc_tmp,R_pc)
    R_grd = R_grd.reshape(batch,k,height,width)
    return R_grd

def pc_to_grid_nearest_id(R_pc,XY_pc,XY_grd,index,interpolator_back):
    # convert pc to grid
    # R_pc: point cloud value with [batch,channels,N] dim
    # XY_pc: point cloud position with [batch,2,N] dim
    #        scaled to [0,1]
    batch,_,height,width = XY_grd.shape
    _,k,_ = R_pc.shape
    XY_grd_tmp = XY_grd.reshape(batch,2,height*width).permute(0,2,1).detach()
    XY_pc_tmp = XY_pc.permute(0,2,1).detach()
    #R_grd = nearest_neighbor_interp_kd(XY_grd_tmp,XY_pc_tmp,R_pc)
    ##R_grd = nearest_neighbor_interp_fe(XY_grd_tmp,XY_pc_tmp,R_pc)
    R_grd = interpolator_back(XY_grd_tmp,XY_pc_tmp,R_pc)
    ###R_grd = nearest_neighbor_interp_fi(XY_grd_tmp,XY_pc_tmp,R_pc,index,"backward")
    R_grd = R_grd.reshape(batch,k,height,width)
    return R_grd

