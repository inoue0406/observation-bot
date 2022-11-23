"""Module summary.

This is predictor.py.
"""

from torch import nn
import torch 
import torchvision
import numpy as np

from models.interp_funcs import pc_to_grid,pc_to_grid_nearest,pc_to_grid_nearest_id

class predictor_interp2d(nn.Module):
    """Summary line.

    This type of predictor performs non-parameterized 2-d interpolation to
    estimate underlying input field.
    """

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
 
    def forward(self, R_pc, XY_pc, XY_grd):
        """Forward. 

        Args:
            R_pc (torch.Tensor): Field value at locations specified by XY_pc with 
                                  [batch,channels,N] dimensions.
            XY_pc (torch.Tensor): The 2-d location of observation bots with [batch,2,N] dimensions,
                                   where N is the number of bots.
            xy_grd (torch.Tensor) : The regular grid with [x,y,2] dimensions.

        Returns:
            R_grd (torch.Tensor): Interpolated field with [batch,channels,height,width] dimensions.

        """

        if self.interp_type == "bilinear":
            R_grd = pc_to_grid(R_pc,XY_pc,height)
        elif self.interp_type == "nearest_normal" or self.interp_type == "nearest_kdtree":
            R_grd = pc_to_grid_nearest(R_pc,XY_pc,XY_grd,self.interp_type,self.interpolator)
        elif self.interp_type == "nearest_faiss":
            R_grd = pc_to_grid_nearest_id(R_pc,XY_pc,XY_grd,self.faiss_index,self.interpolator_back)

        return R_grd

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class predictor_deconv(nn.Module):
    """Summary line.
    This type of predictor performs deconvolution operation from
    observed quantities to estimate the spatial distribution of underlying field.
    """

    def __init__(self, hidden_dim, npc):
        """Initialization.

        Args:
            hidden_dim (int): dimension of hidden representation obtained by conv layers
            npc (int): dimension of point cloud 
        """        
        super(predictor_deconv, self).__init__()
        self.hidden_dim = hidden_dim
        self.npc = npc
        nc = 1
        nf = 32
        self.upc0 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(hidden_dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )

        # without skip connection
        # state size. (nf*8) x 4 x 4
        self.upc1 = dcgan_upconv(nf * 8, nf * 8)
        # state size. (nf*8) x 8 x 8
        self.upc2 = dcgan_upconv(nf * 8, nf * 4)
        # state size. (nf*4) x 16 x 16
        self.upc3 = dcgan_upconv(nf * 4, nf * 2)
        # state size. (nf*2) x 32 x 32
        self.upc4 = dcgan_upconv(nf * 2, nf)
        # state size. (nf) x 64 x 64
        self.upc5 = dcgan_upconv(nf, nf)
        # state size. (nf) x 128 x 128
        self.upc6 = nn.Sequential(
            nn.ConvTranspose2d(nf, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 256 x 256
            )
        
        # FC network for predicting observed quantities
        # fc0 (npc*3) -> (hidden_dim)
        self.fc0 =  nn.Sequential(
                        nn.Linear(self.npc*3, self.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2))
        # fc1 (hidden_dim) -> (hidden_dim)
        self.fc1 =  nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2))
        # fc1 (hidden_dim) -> (hidden_dim)
        self.fc2 =  nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2))

    def forward(self, R_pc, XY_pc, XY_grd):
        """Forward. 

        Args:
            R_pc (torch.Tensor): Field value at locations specified by XY_pc with 
                                  [batch,channels,N] dimensions.
            XY_pc (torch.Tensor): The 2-d location of observation bots with [batch,2,N] dimensions,
                                   where N is the number of bots.
            xy_grd (torch.Tensor) : The regular grid with [x,y,2] dimensions.

        Returns:
            R_grd (torch.Tensor): Interpolated field with [batch,channels,height,width] dimensions.

        """
        # FC network
        hidden = torch.cat([R_pc,XY_pc],1)
        h_fc0 = self.fc0(hidden.view(-1,self.npc*3))
        h_fc1 = self.fc1(h_fc0)
        h_fc2 = self.fc2(h_fc1)

        # Deconv network
        # no skip connection
        d0 = self.upc0(h_fc2.view(-1, self.hidden_dim, 1, 1))
        d1 = self.upc1(d0)
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        d5 = self.upc5(d4)
        output = self.upc6(d5)
        return output

        return None

class predictor_convlstm(nn.Module):
    """Summary line.

    This type of predictor performs spatio-temporal prediction from 
    observed quantities to account for time dependencies.
    (Currently, this class is not implemented)
    """

    def __init__(self, ):
        super().__init__()
 
    def forward(self, input):
        """sum 2 values.

        Args:
            x (float): 1st argument
            y (float): 2nd argument

        Returns:
            float: summation

        """
        
        bsize, tsize, channels, height, width = input.size()

        # Lagrangian prediction
        R_grd = input[:,0,:,:,:] #use initial

        return None
