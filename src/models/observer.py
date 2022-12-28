"""Module summary.

This is observer.py.
"""

from torch import nn
import torch
import numpy as np

# Import interpolation tool.
from models.interp_funcs import grid_to_pc,grid_to_pc_nearest,grid_to_pc_nearest_id

class observer_interp2d(nn.Module):
    """Summary line.

    This is the "interpolation type" observer, which obtaines observed quantities
    by performing interpolation at specified locations.
    This observer does NOT have trainable parameters. 
    """
    # Main Class for the Observation Bot

    def __init__(self, interp_type):
        """Initialization.

        Args:
            interp_type (str): type of interpolation algorithm
        """
        super().__init__()
        self.interp_type = interp_type
        # import according to interp type
        if interp_type == "nearest_normal":
            from nearest_neighbor_interp_normal import nearest_neighbor_interp
            self.interpolator = nearest_neighbor_interp
        elif interp_type == "nearest_kdtree":
            from nearest_neighbor_interp_kdtree import nearest_neighbor_interp_kd
            self.interpolator = nearest_neighbor_interp_kd
 
    def forward(self, R, XY_pc, XY_grd):
        """Forward.

        Args:
            R (torch.Tensor): The ground truth input field with [batch,channels,height,width] dimensions.
            XY_pc (torch.Tensor): The 2-d location of observation bots with [batch,2,N] dimensions,
                                   where N is the number of bots.
            xy_grd (torch.Tensor) : The regular grid with [x,y,2] dimensions.

        Returns:
            R_pc (torch.Tensor): Interpolated field value at locations specified by XY_pc with 
                                  [batch,channels,N] dimensions.

        """

        if self.interp_type == "bilinear":
            R_pc = grid_to_pc(R[:,:,:,:],XY_pc)
        elif self.interp_type == "nearest_normal" or self.interp_type == "nearest_kdtree":
            R_pc = grid_to_pc_nearest(R[:,:,:,:],XY_pc,XY_grd,self.interp_type,self.interpolator)
        elif self.interp_type == "nearest_faiss":
            R_pc = grid_to_pc_nearest_id(R[:,:,:,:],XY_pc,XY_grd,self.faiss_index,self.interpolator_fwd)

        return R_pc

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class observer_conv(nn.Module):
    """Summary line.

    This is a "CNN" type observer, which obtaines observed quantities
    by performing convolution operations to the true field.
    """
    # Main Class for the Observation Bot

    def __init__(self, hidden_dim, npc):
        """Initialization.

        Args:
            hidden_dim (int): dimension of hidden representation obtained by conv layers
            npc (int): dimension of point cloud 
        """
        super(observer_conv, self).__init__()
        self.hidden_dim = hidden_dim
        self.npc = npc
        nc = 1
        nf = 32
        # input is (nc) x 256 x 256
        self.c0 = dcgan_conv(nc, nf)
        # input is (nf) x 128 x 128
        self.c1 = dcgan_conv(nf, nf)
        # state size. (nf) x 64 x 64
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c6 = nn.Sequential(
                nn.Conv2d(nf * 8, hidden_dim, 4, 1, 0),
                nn.BatchNorm2d(hidden_dim),
                nn.Tanh()
                )
        
        # FC network for predicting observed quantities
        r_drop = 0.0
        # fc0 (npc*2) -> (hidden_dim)
        self.fc0 =  nn.Sequential(
                        nn.Linear(self.npc*2, self.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(r_drop))
        # fc1 (hidden_dim*2) -> (hidden_dim)
        self.fc1 =  nn.Sequential(
                        nn.Linear(self.hidden_dim*2, self.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(r_drop))
        self.fc2 =  nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(r_drop))
        self.fc3 =  nn.Sequential(
                        nn.Linear(self.hidden_dim, self.npc),
                        nn.ReLU(),
                        nn.Dropout(r_drop))

    def forward(self, R, XY_pc, XY_grd):
        """sum 2 values.

        Args:
            R (torch.Tensor): The ground truth input field with [batch,channels,height,width] dimensions.
            XY_pc (torch.Tensor): The 2-d location of observation bots with [batch,2,N] dimensions,
                                   where N is the number of bots.
            xy_grd (torch.Tensor) : The regular grid with [x,y,2] dimensions.

        Returns:
            R_pc (torch.Tensor): Interpolated field value at locations specified by XY_pc with 
                                  [batch,channels,N] dimensions.

        """

        # The conv part
        h0 = self.c0(R)
        h1 = self.c1(h0)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        h6 = self.c6(h5)
        # no skip connection
        h_conv = h6.view(-1, self.hidden_dim)
        
        # predicting point observation
        # (fully-connected network)
        
        h_xy = self.fc0(XY_pc.view(-1,self.npc*2))
        # combine convnet output and XY information
        h_fc1 = self.fc1(torch.cat([h_conv,h_xy],1))
        h_fc2 = self.fc2(h_fc1)
        R_pc = self.fc3(h_fc2)
        # adjust shape
        R_pc = R_pc[:,None,:]
        return R_pc
