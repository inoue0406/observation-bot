"""Module summary.

This is predictor.py.
"""

import torch 
import torchvision
import numpy as np

from interp_funcs import pc_to_grid,pc_to_grid_nearest,pc_to_grid_nearest_id

class predictor_interp2d(nn.Module):
    """Summary line.

    This type of predictor performs non-parameterized 2-d interpolation to
    estimate underlying input field.
    """

    def __init__(self, ):
        super().__init__()
        self.interp_type = interp_type
        # import according to interp type
        if interp_type == "nearest_normal":
            from nearest_neighbor_interp_normal import nearest_neighbor_interp
            self.interpolator = nearest_neighbor_interp
        elif interp_type == "nearest_kdtree":
            from nearest_neighbor_interp_kdtree import nearest_neighbor_interp_kd
            self.interpolator = nearest_neighbor_interp_kd
 
    def forward(self, R_pc, XY_pc):
        """Forward.

        Args:
            R_pc (torch.Tensor): Field value at locations specified by XY_pc with 
                                  [batch,channels,N] dimensions.
            XY_pc (torch.Tensor): The 2-d location of observation bots with [batch,2,N] dimensions,
                                   where N is the number of bots.

        Returns:
            R_grd (torch.Tensor): Interpolated field with [batch,channels,height,width] dimensions.

        """

        if self.interp_type == "bilinear":
            R_grd = pc_to_grid(R_pc,XY_pc,height)
        elif self.interp_type == "nearest_normal" or self.interp_type == "nearest_kdtree":
            R_grd = pc_to_grid_nearest(R_pc,XY_pc,XY_grd,self.interp_type,self.interpolator)
        elif self.interp_type == "nearest_faiss":
            R_grd = pc_to_grid_nearest_id(R_pc,XY_pc,XY_grd,self.faiss_index,self.interpolator_back)


        # Lagrangian prediction
        R_grd = input[:,0,:,:,:] #use initial


class predictor_deconv(nn.Module):
    """Summary line.

    This type of predictor performs deconvolution operation from
    observed quantities to estimate the spatial distribution of underlying field.
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
