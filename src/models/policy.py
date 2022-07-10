"""Module summary.

This is policy.py.
"""

from torch import nn
import torch
import numpy as np

class policy_obsbot(nn.Module):
    """Summary line.

    This is policy_obsbot.
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
