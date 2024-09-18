"""
Computes the energy and score of a given image using the energy model proposed in 
Multi-Scale Energy (MuSE) Framework for Inverse Problems in Imaging:
I_theta(x) = 0.5 *||x-psi(x)||^2 

Jyothi Rikhab Chand, 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv2d, conv_transpose2d, linear
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


class EBM(nn.Module):
    """
    Summary of class EBM here:
    Attributes: net: psi(x) C^m -> C^m
    Example usage: 
    energy_net = EBM(net)      
    Energy_value = energy_net(x)
    Energy_value, score = energy_net.giveScore(x)    
    """

    def __init__(self,net):
        """
        Initializes an EBM object.

        Parameters:
            net: psi(x) C^m -> C^m    
        """
        super().__init__()
        self.net = net
     
    def forward(self, input):
        """
        Computes the energy value of the given input
        
        Parameters: 
               input: complex image whose energy value
                      has to be computed
        Returns:
                E: 0.5 ||y-net(input)||^2
        """
        
        x = torch.cat((input.real,input.imag),dim=1) 
        y = self.net(x)
        E= 0.5*torch.sum((y-x).abs()**2,dim =(1,2,3)) 
        return E

    def giveScore(self,input):
        """
        Computes the energy value and score of the given input
        
        Parameters: 
               input: complex image whose energy value
                      & sore has to be computed
        Returns:
                E: 0.5 ||y-net(input)||^2 & score: gradient of E w.r.t x computed using autograd function
        """
        
        x = torch.cat((input.real,input.imag),dim=1) 
        x=x.requires_grad_(True)
        y= self.net(x)
        E = 0.5*torch.sum((y-x).abs()**2,dim =(1,2,3))
        score =torch.autograd.grad(E, x, grad_outputs= torch.ones_like(E), create_graph=True, only_inputs=True)[0]
        return E, score[:,0:1]+1j*score[:,1:2]