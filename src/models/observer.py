"""Module summary.

This is observer.py.
"""

from torch import nn
import torch
import numpy as np

class observer_obsbot_interp_2d(nn.Module):
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
 
    def forward(self, R, XY_pc):
        """sum 2 values.

        Args:
            R (torch.Tensor): The ground truth input field with [batch,channels,height,width] dimensions.
            XY_pc (torch.Tensor): The 2-d location of observation bots with [batch,2,N] dimensions,
                                   where N is the number of bots.

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

