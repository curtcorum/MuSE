"""
Implements the accelerated MM algorithm proposed in  Multi-Scale Energy (MuSE) Framework for 
Inverse Problems in Imaging
 
Jyothi Rikhab Chand, 2024
"""



import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def posterior_grad(energy_net,A,x_iterate,b,tstCsm,tstMask,inference_std,score_std):
    """
    Computes the gradient w.r.t x_iterate of the modified posterior term
    Parameters:
    energy_net: pre-trained energy network
    A: MRI forward operator
    x_iterate: iterate w.r.t which gradient has to be computed of the modified posterior term
    b: Undersampled data
    tstCsm: Coil sensitivity map
    tstMask: Mask
    Returns:
    grad: gradient of modified posterior w.r.t x_iterate
    likelihood: value of the modified likelihood
    energy: energy value at x_iterate
   
    """
    x_iterate=x_iterate.requires_grad_(True)
    #computation of energy and score w.r.t. x_iterate
    energy,score = energy_net.giveScore(x_iterate)
        
        
    #computation of new likelihood
    dc_adj=A.forward(x_iterate-score,tstCsm,tstMask)
    likelihood_adj = (torch.sum((dc_adj-b).abs()**2,dim =(1,2,3)))/(2*inference_std**2)
        
    #gradient of likelihood w.r.t x_iterate
    grad_likelihood =torch.autograd.grad(likelihood_adj , x_iterate , grad_outputs= torch.ones_like(likelihood_adj), create_graph=True, only_inputs=True)[0]
        
    #gradient of posterior w.r.t x_iterate
    gradient = grad_likelihood  + (score/(score_std**2))
        
    x_iterate = x_iterate.requires_grad_(False)
    del x_iterate
                                             
    energy = energy.detach()
    likelihood_adj=likelihood_adj.detach()
    grad_likelihood = grad_likelihood.detach()
    gradient= gradient.detach()
        
    del score,grad_likelihood
    return gradient,likelihood_adj,energy
    
        


   

    
#------------------------------------------------------------------------------------ 
def MM(x_init,energy_net,A,b, tstOrg,tstCsm,tstMask,Atb, inference_std,score_std,L,alpha,max_iter,threshold):
    """
    Implements the accelerated MM algorithm.
    Parameters:
    x_init: initial guess
    energy_net: pre-trained energy network
    A: MRI forward operator
    b: Undersampled data
    tstOrg: Org image
    tstCsm: Coil sensitivity map
    tstMask: Mask
    Atb: ATb
    L: Lipschitz
    max_iter: maximum iterations
    threshold: exit condition
    Returns:
    x_iter: converged solution
    """

    obj_mm=[]
    x_iter=x_init 
    const = 2*(inference_std**2)
    for itr in range(max_iter):
    
        x_clone = x_iter.clone()
        grad,log_likelihood,energy=posterior_grad(energy_net,A,x_clone,b,tstCsm,tstMask,inference_std,score_std)
        del x_clone
        
        obj_val = log_likelihood+(energy/(score_std**2))
        obj_mm.append(obj_val)
        if itr>=1:
            if torch.abs((obj_mm[itr] - obj_mm[itr-1])/obj_mm[itr-1])<=threshold:
                break
        
        #solves the surrogate minimization of MM with modified posterior
        #rhs is for CG.
        rhs = (((alpha**2)*A.ATA(x_iter,tstCsm,tstMask))/const) - grad +((L*x_iter)/score_std**2)
        x_iter = A.mm_inv(x_iter,rhs,tstCsm,tstMask,(L/(score_std**2)),((alpha**2)/const))
        
        
        
    return x_iter


#----------------------------------------------------------------------------------------