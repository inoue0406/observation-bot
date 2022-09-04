"""Module summary.

This is predictor.py.
"""

import torch 
import torchvision
import numpy as np

class predictor_interp2d(nn.Module):
    """Summary line.

    This type of predictor performs non-parameterized 2-d interpolation to
    estimate underlying input field.
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


