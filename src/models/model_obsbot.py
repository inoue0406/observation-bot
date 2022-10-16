from torch import nn
import torch
import numpy as np

# Import Observer/Policy/Predictor classes.
from models.observer import observer_interp2d
from models.policy import policy_lstm
from models.predictor import predictor_interp2d

class obsbot(nn.Module):
    # Main Class for the Observation Bot

    def __init__(self, image_size, pc_size, batch_size,
                 mode="run", interp_type="bilinear"):
        super().__init__()

        # set regular grid
        self.Xgrid,self.Ygrid = self.xy_grid(image_size,image_size)
        # set pc grid
        self.Xpc,self.Ypc = self.xy_grid(pc_size,pc_size)
        # number of point cloud
        self.npc = pc_size*pc_size
        # mode
        self.mode = mode
        self.interp_type = interp_type

        # Initialize observer/policy/predictor networks
        self.observer = observer_interp2d(interp_type)
        self.policy = policy_lstm(pc_size, self.npc)
        self.predictor = predictor_interp2d(interp_type)

    def xy_grid(self,height,width):
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