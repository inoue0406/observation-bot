import torch

import numpy as np

def batch_pairwise_distances(x, y=None):
    '''
    Input: x is a bxNxd matrix where b is batch size
           y is an optional bxMxd matirx
    Output: dist is a bxNxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    b,N,d = x.shape
    x_norm = (x**2).sum(2).view(b,-1, 1)
    if y is not None:
        y_norm = (y**2).sum(2).view(b, 1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
        
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, torch.transpose(y, 1, 2))
    return dist

def nearest_neighbor_interp(xy_grd_b,xy_pc_b,R_pc_b):
    '''
    Input: xy_grd_b bxNx2 matrix where b is batch size, N is regular grid mesh size
           xy_pc_b  bxMx2 matrix where b is batch size, M is point cloud size
           R_pc_b   bxkxM matrix where b is batch size, M is point cloud size
    Output: R_grd_b  bxkxN interpolated value at grid point
    '''
    b,k,M = R_pc_b.shape
    _,N,_ = xy_grd_b.shape
    #D = batch_pairwise_distances(xy_grd_b, xy_pc_b)
    #id_min = torch.min(D,2).indices
    id_min = torch.zeros([b,N],dtype=torch.int64).cuda()
    #splits = 1000
    splits = 100
    #for n in range(N):
    #    D = batch_pairwise_distances(xy_grd_b[:,n:n+1,:], xy_pc_b)
    #    id_min[:,n:n+1] = torch.min(D,2).indices
    for nsp in np.array_split(range(N),splits):
        D = batch_pairwise_distances(xy_grd_b[:,nsp,:], xy_pc_b)
        id_min[:,nsp] = torch.min(D,2).indices
        #if(n % 1000 == 0):
        #    print("n=",n)
    # add dimension 
    #id_min = id_min[:,None,:]
    id_min = torch.stack(k*[id_min],axis=1)
    # interpolate from point cloud to grid
    R_grd_b = torch.gather(R_pc_b,2,id_min)
    return R_grd_b

if __name__ == '__main__':
    # test for distance function
    # define size
    height = 50
    width = 50
    # define position of circle
    ic = 25
    jc = 25
    # spatial scale
    scale = 5

    # create initial field
    R = torch.zeros(1,1,height,width)
    # create xy grid
    xx = torch.arange(0, width).view(1, -1).repeat(height, 1).float()
    yy = torch.arange(0, height).view(-1, 1).repeat(1, width).float()
    # create circular field
    R[0,0,:,:] = torch.exp(-((xx-ic)**2 + (yy-jc)**2)/scale**2)
    
    # Define point cloud with 2d initialization
    # point cloud size
    Npc = 1600
    # torch.rand generates uniform [0,1)
    XY = torch.rand(1, 2, Npc, dtype=torch.float32, requires_grad=True)
    XY = XY * height # Scale to uniform [0,height]
    print("xy shape",XY.shape)

    R = torch.zeros(1,1,Npc)
    # create circular field
    R[0,0,:] = torch.exp(-((XY[0,0]-ic)**2 + (XY[0,1]-jc)**2)/scale**2)

    # scale to [-1,1]
    #XYZ_scl = XYZ/25.0*2.0-1.0
    XY_scl = XY/50.0
    print(torch.min(XY_scl),torch.max(XY_scl))

    # Regular Grid: xx,yy [50,50]
    # Irregular Grid: XY [1,2,1600]
    xy_grd = torch.stack([xx,yy]).reshape(2,50*50).permute(1,0)
    print("shape of xy_grd:",xy_grd.shape)
    xy_pc = XY[0,:,:].permute(1,0)
    print("shape of xy_pc:",xy_pc.shape)
    # make example of batch size 2
    xy_grd_b = torch.stack([xy_grd,xy_grd])
    xy_pc_b = torch.stack([xy_pc,xy_pc])
    R_pc_b = torch.cat([R,R])
    RR_pc_b = torch.cat(3*[R_pc_b],dim=1)
    print("shape of xy_p_bc:",xy_pc_b.shape)

    #R_grd_b = nearest_neighbor_interp(xy_grd_b,xy_pc_b,R_pc_b)

    R_grd_b = nearest_neighbor_interp(xy_grd_b,xy_pc_b,RR_pc_b)
    
    #R_grd_b2 = nearest_neighbor_interp_kd(xy_grd_b,xy_pc_b,RR_pc_b)
    
    import pdb;pdb.set_trace()
    
