from torch import nn
import torch

import numpy as np

def xy_grid(height,width):
    # generate constant xy grid
    x1grd = torch.linspace(0,1,width).cuda() # 1d grid
    y1grd = torch.linspace(0,1,height).cuda() # 1d grid

    Xgrid = torch.zeros(height, width)
    Ygrid = torch.zeros(height, width)
    for j in range(height):
        Xgrid[j,:] = x1grd
    for k in range(width):
        Ygrid[:,k] = y1grd
    
    return Xgrid,Ygrid

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

class LSTMcell(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout,
                           batch_first=True)

        # linear layer for converting hidden dim to output dim
        self.fc_out = nn.Linear(hid_dim,output_dim)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, in_seq):
            
        #in_seq = [in_seq len, batch size]
        
        in_seq = self.dropout(in_seq)
        
        rnn_out, (hidden, cell) = self.rnn(in_seq)
        
        #outputs = [in_seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]

        outputs = self.fc_out(rnn_out)
        
        #outputs are always from the top hidden layer

        return outputs

class obsbot_seq2seq(nn.Module):
    # Main Class for the Observation Bot

    def __init__(self, lstm, image_size, pc_size, batch_size,
                 mode="run", interp_type="bilinear"):
        super().__init__()
        # set regular grid
        self.Xgrid,self.Ygrid = xy_grid(image_size,image_size)
        # set pc grid
        self.Xpc,self.Ypc = xy_grid(pc_size,pc_size)
        # number of point cloud
        self.npc = pc_size*pc_size
        # mode
        self.mode = mode
        self.interp_type = interp_type
        # lstm
        self.lstm = lstm
        # import according to interp type
        if interp_type == "nearest_normal":
            from nearest_neighbor_interp_normal import nearest_neighbor_interp
            self.interpolator = nearest_neighbor_interp
        elif interp_type == "nearest_kdtree":
            from nearest_neighbor_interp_kdtree import nearest_neighbor_interp_kd
            self.interpolator = nearest_neighbor_interp_kd
        elif interp_type == "nearest_faiss":
            from nearest_neighbor_interp_faiss import nearest_neighbor_interp_fe,nearest_neighbor_interp_fi,set_index_faiss
            from nearest_neighbor_interp_kdtree import nearest_neighbor_interp_kd
            self.interpolator_fwd = nearest_neighbor_interp_fi
            #self.interpolator_back = nearest_neighbor_interp_fe
            self.interpolator_back = nearest_neighbor_interp_kd
            # set FAISS index
            X_grd = torch.stack(batch_size*[self.Xgrid]).unsqueeze(1)
            Y_grd = torch.stack(batch_size*[self.Ygrid]).unsqueeze(1)
            XY_grd = torch.cat([X_grd,Y_grd],dim=1).cuda()
            points_grd = XY_grd[0,:,:].reshape(2,image_size*image_size).permute(1,0).detach().cpu().numpy()
            self.faiss_index = set_index_faiss(points_grd)

    def forward(self, input):
        
        # Grid variable: XY_grd, R_grd
        # Point cloud variable: XY_pc, R_pc
        
        bsize, tsize, channels, height, width = input.size()

        # Lagrangian prediction
        R_grd = input[:,0,:,:,:] #use initial

        # Set Initial Grid (which will be fixed through time progress)
        X_grd = torch.stack(bsize*[self.Xgrid]).unsqueeze(1)
        Y_grd = torch.stack(bsize*[self.Ygrid]).unsqueeze(1)
        XY_grd = torch.cat([X_grd,Y_grd],dim=1).cuda()
        # Set Initial PC
        X_pc = torch.stack(bsize*[self.Xpc]).unsqueeze(1)
        Y_pc = torch.stack(bsize*[self.Ypc]).unsqueeze(1)
        XY_pc = torch.cat([X_pc,Y_pc],dim=1).cuda()
        XY_pc = XY_pc.clone().reshape(bsize,2,self.npc)

        xout = torch.zeros(bsize, tsize, channels, height, width,  requires_grad=True).cuda()
        if self.mode == "check":
            r_pc_out = torch.zeros(bsize, tsize, channels, self.npc).cuda()
            xy_pc_out = torch.zeros(bsize, tsize, 2, self.npc).cuda()
        
        for it in range(tsize):
            # -----------------------------------------------
            # (1) Observation: Interpolate UV to Point Cloud position
            if self.interp_type == "bilinear":
                R_pc = grid_to_pc(input[:,it,:,:,:],XY_pc)
            elif self.interp_type == "nearest_normal" or self.interp_type == "nearest_kdtree":
                R_pc = grid_to_pc_nearest(input[:,it,:,:,:],XY_pc,XY_grd,self.interp_type,self.interpolator)
            elif self.interp_type == "nearest_faiss":
                R_pc = grid_to_pc_nearest_id(input[:,it,:,:,:],XY_pc,XY_grd,self.faiss_index,self.interpolator_fwd)
                
            # -----------------------------------------------
            # (2) Motion Estimator: Predict Single Time Step with RNNs
            # R_pc,hidden = Predict(XY_pc,R_pc,hidden)
            XYR_pc = torch.cat([XY_pc,R_pc],dim=1).reshape(bsize,self.npc*3)
            dXY = self.lstm(XYR_pc)
            XY_pc = XY_pc + dXY.reshape(bsize,2,self.npc)

            # -----------------------------------------------
            # (3) Field Estimator : Estimate Field Value from point observation
            if self.interp_type == "bilinear":
                R_grd = pc_to_grid(R_pc,XY_pc,height)
            elif self.interp_type == "nearest_normal" or self.interp_type == "nearest_kdtree":
                R_grd = pc_to_grid_nearest(R_pc,XY_pc,XY_grd,self.interp_type,self.interpolator)
            elif self.interp_type == "nearest_faiss":
                R_grd = pc_to_grid_nearest_id(R_pc,XY_pc,XY_grd,self.faiss_index,self.interpolator_back)

            xout[:,it,:,:,:] = R_grd
            if self.mode == "check":
                r_pc_out[:,it,:,:] = R_pc
                xy_pc_out[:,it,:,:] = XY_pc

        if self.mode == "run":
            return xout
        elif self.mode == "check":
            return xout,r_pc_out,xy_pc_out
