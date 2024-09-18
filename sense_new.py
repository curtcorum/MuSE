import numpy as np
import torch.nn as nn
import torch

#@title Define A and adjoint of A operators
class cg_block(nn.Module):
    def __init__(self, cgIter, cgTol):
        super(cg_block, self).__init__()
        self.cgIter = cgIter
        self.cgTol = cgTol
        
    def forward(self, lhs, rhs, x0):
        fn=lambda a,b: torch.abs(torch.sum(torch.conj(a)*b,axis=[-1,-2,-3]))
        x = x0
        r = rhs-lhs(x0)
        p = r
        rTr = fn(r,r)
        eps=torch.tensor(1e-10)
        for i in range(self.cgIter):
            Ap = lhs(p)
            alpha=rTr/(fn(p,Ap)+eps)
            x = x +  alpha[:,None,None,None] * p
            r = r -  alpha[:,None,None,None] * Ap
            rTrNew = fn(r,r)
            if torch.sum(torch.sqrt(rTrNew+eps)) < self.cgTol:
                
                break
                
            beta = rTrNew / (rTr+eps)
            p = r + beta[:,None,None,None] * p
            rTr=rTrNew
           
        return x



class sense_v1(nn.Module):
    def __init__(self, cgIter):
        super().__init__()
        
        self.cgIter = cgIter
        self.cg = cg_block(self.cgIter, 1e-3)#1e-12
        

    def forward(self, img, csm, mask):
        cimg = img*csm
        cimg = torch.fft.fftshift(cimg,dim=[-1,-2])
        mcksp = torch.fft.fft2(cimg,dim=[-1,-2],norm="ortho")
        mcksp = torch.fft.ifftshift(mcksp,dim=[-1,-2])
        usksp = mcksp * mask
        return usksp
        
    def adjoint(self, ksp, csm):
        ksp = torch.fft.fftshift(ksp,dim=[-1,-2])
        img = torch.fft.ifft2(ksp,dim=[-1,-2],norm="ortho")
        img = torch.fft.ifftshift(img,dim=[-1,-2])
        cs_weighted_img = torch.sum(img*torch.conj(csm),1,True)
        return cs_weighted_img
    
    #def adjoint_lipschitz(self, ksp, csm, mask):
    #    ksp = ksp*mask
    #    img = torch.fft.ifft2(ksp,dim=[-1,-2],norm="ortho")
    #    cs_weighted_img = torch.sum(img*torch.conj(csm),1,True)
    #    return cs_weighted_img
    
    def ATA(self, img, csm, mask):
        cimg = img*csm
        cimg = torch.fft.fftshift(cimg,dim=[-1,-2])
        mcksp = torch.fft.fft2(cimg,dim=[-1,-2],norm="ortho")
        mcksp = torch.fft.ifftshift(mcksp,dim=[-1,-2])
        usksp = mcksp * mask
        usksp = torch.fft.fftshift(usksp,dim=[-1,-2])
        usimg = torch.fft.ifft2(usksp,dim=[-1,-2],norm="ortho")
        usimg = torch.fft.ifftshift(usimg,dim=[-1,-2])
        cs_weighted_img = torch.sum(usimg*torch.conj(csm),1,True)
        return cs_weighted_img
    
    
    def mm_inv(self, x0, rhs, csm, mask,lipschitz_bound,lam):
        
        lhs = lambda x: (lam)*self.ATA(x, csm, mask) + lipschitz_bound*x 
        out = self.cg(lhs, rhs, x0)
        
        return out
    
    def sense_sol(self, x0, rhs, lam, csm, mask):
        
        lhs = lambda x: lam*self.ATA(x, csm, mask) + 1.000*x
        out = self.cg(lhs, lam*rhs, x0)
        
        return out
    
    def ls_sol(self, x0, rhs, csm, mask):
        
        lhs = lambda x: self.ATA(x, csm, mask) 
        out = self.cg(lhs, rhs, x0)
        
        return out