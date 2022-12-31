from torch import nn
import torch
import numpy as np

# Import Observer/Policy/Predictor classes.
from models.observer import observer_interp2d, observer_conv
from models.policy import policy_lstm
from models.predictor import predictor_interp2d, predictor_deconv

class obsbot(nn.Module):
    # Main Class for the Observation Bot

    def __init__(self, image_size, pc_size, batch_size,
                 mode="run", observer_type="interp2d",policy_type="seq2seq",predictor_type="interp2d",
                 freeze=[0,0,0],observer_transfer_path=None,
                 interp_type="bilinear", pc_initialize="regular"):
        super().__init__()

        # set regular grid
        self.Xgrid,self.Ygrid = self.xy_grid(image_size,image_size)
        # set pc grid
        if pc_initialize=="regular":
            # regular grid
            self.Xpc,self.Ypc = self.xy_grid(pc_size,pc_size)
            # 2d to 1d variable
            self.Xpc = self.Xpc.flatten()
            self.Ypc = self.Ypc.flatten()
        elif pc_initialize=="random":
            # random point initialization
            self.Xpc = torch.rand(pc_size*pc_size)
            self.Ypc = torch.rand(pc_size*pc_size)

        # number of point cloud
        self.npc = pc_size*pc_size
        # mode
        self.mode = mode
        self.interp_type = interp_type

        # Initialize observer/policy/predictor networks
        # Observer Network
        if observer_type == "interp2d":
            self.observer = observer_interp2d(interp_type)
        elif observer_type == "conv2d":
            hidden_dim = 128
            self.observer = observer_conv(hidden_dim,self.npc)
        # Policy Network
        if policy_type == "seq2seq":
            self.policy = policy_lstm(pc_size, self.npc)
        # Predictor Network
        if predictor_type == "interp2d":
            self.predictor = predictor_interp2d(interp_type)
        elif predictor_type == "deconv2d":
            hidden_dim = 128
            self.predictor = predictor_deconv(hidden_dim,self.npc)

        # Freeze weights for each type of network
        if freeze[0] == 1:
            # Freeze Observer
            for param in self.observer.parameters():
                param.requires_grad = False
        if freeze[1] == 1:
            # Freeze Policy
            for param in self.policy.parameters():
                param.requires_grad = False
        if freeze[2] == 1:
            # Freeze Predictor
            for param in self.predictor.parameters():
                param.requires_grad = False

        # Use pretrained weights for transfer learning
        if observer_transfer_path != None:
            print('loading pretrained model:',observer_transfer_path)
            # Load from state dictionary
            import pdb;pdb.set_trace()
            self.observer.load_state_dict(torch.load(observer_transfer_path))

    def xy_grid(self,height,width):
        # generate constant xy grid in [0,1] range
        x1grd = torch.linspace(0,1,width).cuda() # 1d grid
        y1grd = torch.linspace(0,1,height).cuda() # 1d grid

        Xgrid = torch.zeros(height, width)
        Ygrid = torch.zeros(height, width)
        for j in range(height):
            Xgrid[j,:] = x1grd
        for k in range(width):
            Ygrid[:,k] = y1grd
    
        return Xgrid,Ygrid

    def forward(self, input):
        
        # Grid variable: XY_grd, R_grd
        # Point cloud variable: XY_pc, R_pc
        
        bsize, tsize, channels, height, width = input.size()

        R_grd = input[:,0,:,:,:] #use initial

        # Set Initial Grid (which will be fixed through time progress)
        X_grd = torch.stack(bsize*[self.Xgrid]).unsqueeze(1)
        Y_grd = torch.stack(bsize*[self.Ygrid]).unsqueeze(1)
        XY_grd = torch.cat([X_grd,Y_grd],dim=1).cuda()
        # Set Initial PC
        X_pc = torch.stack(bsize*[self.Xpc]).unsqueeze(1)
        Y_pc = torch.stack(bsize*[self.Ypc]).unsqueeze(1)
        XY_pc = torch.cat([X_pc,Y_pc],dim=1).cuda()

        xout = torch.zeros(bsize, tsize, channels, height, width,  requires_grad=True).cuda()
        if self.mode == "check":
            r_pc_out = torch.zeros(bsize, tsize, channels, self.npc).cuda()
            xy_pc_out = torch.zeros(bsize, tsize, 2, self.npc).cuda()
        
        for it in range(tsize):
            # ----------------------------------------------------------
            # (1) Observation: Interpolate UV to Point Cloud position.
            R_pc = self.observer(input[:,it,:,:,:],XY_pc,XY_grd) 

            # ----------------------------------------------------------
            # (2) Field Estimator : Estimate Field Value from point observation.
            R_grd = self.predictor(R_pc,XY_pc,XY_grd)

            # ----------------------------------------------------------
            # (3) Motion Estimator: Predict Single Time Step with RNNs.
            XY_pc = self.policy(XY_pc, R_pc, bsize)

            xout[:,it,:,:,:] = R_grd
            if self.mode == "check":
                r_pc_out[:,it,:,:] = R_pc
                xy_pc_out[:,it,:,:] = XY_pc

        if self.mode == "run":
            return xout
        elif self.mode == "check":
            return xout,r_pc_out,xy_pc_out

class obsbot_observer(nn.Module):
    # A class for the Observer-only runs

    def __init__(self, image_size, pc_size, batch_size,
                 mode="run", observer_type="interp2d",
                 interp_type="bilinear", pc_initialize="regular"):
        super().__init__()

        # set regular grid
        self.Xgrid,self.Ygrid = self.xy_grid(image_size,image_size)
        # set pc grid
        if pc_initialize=="regular":
            # regular grid
            self.Xpc,self.Ypc = self.xy_grid(pc_size,pc_size)
            # 2d to 1d variable
            self.Xpc = self.Xpc.flatten()
            self.Ypc = self.Ypc.flatten()
        elif pc_initialize=="random":
            # random point initialization
            self.Xpc = torch.rand(pc_size*pc_size)
            self.Ypc = torch.rand(pc_size*pc_size)

        # number of point cloud
        self.npc = pc_size*pc_size
        # mode
        self.mode = mode
        self.interp_type = interp_type

        # Observer Network (Teacher)
        self.observer_t = observer_interp2d(interp_type)

        # Observer Network (Models to be trained)
        if observer_type == "interp2d":
            self.observer = observer_interp2d(interp_type)
        elif observer_type == "conv2d":
            hidden_dim = 128
            self.observer = observer_conv(hidden_dim,self.npc)

    def xy_grid(self,height,width):
        # generate constant xy grid in [0,1] range
        x1grd = torch.linspace(0,1,width).cuda() # 1d grid
        y1grd = torch.linspace(0,1,height).cuda() # 1d grid

        Xgrid = torch.zeros(height, width)
        Ygrid = torch.zeros(height, width)
        for j in range(height):
            Xgrid[j,:] = x1grd
        for k in range(width):
            Ygrid[:,k] = y1grd
    
        return Xgrid,Ygrid

    def forward(self, input):
        
        # Grid variable: XY_grd, R_grd
        # Point cloud variable: XY_pc, R_pc
        
        bsize, tsize, channels, height, width = input.size()

        R_grd = input[:,0,:,:,:] #use initial

        # Set Initial Grid (which will be fixed through time progress)
        X_grd = torch.stack(bsize*[self.Xgrid]).unsqueeze(1)
        Y_grd = torch.stack(bsize*[self.Ygrid]).unsqueeze(1)
        XY_grd = torch.cat([X_grd,Y_grd],dim=1).cuda()
        # Set Initial PC
        X_pc = torch.stack(bsize*[self.Xpc]).unsqueeze(1)
        Y_pc = torch.stack(bsize*[self.Ypc]).unsqueeze(1)
        XY_pc = torch.cat([X_pc,Y_pc],dim=1).cuda()

        xout_t = torch.zeros(bsize, tsize, channels, self.npc,  requires_grad=True).cuda()
        xout = torch.zeros(bsize, tsize, channels, self.npc,  requires_grad=True).cuda()

        for it in range(tsize):
            # ----------------------------------------------------------
            # (1) Observation: Interpolate UV to Point Cloud position.
            xout_t[:,it,:,:] = self.observer_t(input[:,it,:,:,:],XY_pc,XY_grd)
            xout[:,it,:,:] = self.observer(input[:,it,:,:,:],XY_pc,XY_grd)

        if self.mode == "run":
            return xout_t,xout
