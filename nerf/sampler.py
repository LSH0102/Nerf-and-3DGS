
import torch
import torch.nn as nn
from rays import Rays

class UniformSampler(nn.Module):
    def __init__(
        self,num_samples,min_depth,max_depth
        
    ):
        super().__init__()
        self.num_samples=num_samples
        self.min_depth=min_depth
        self.max_depth=max_depth
        
    def forward(self,rays):
        Nv=rays.sample_points.shape[0]
        z_vals=self.min_depth+(self.max_depth-self.min_depth)/(self.num_samples+1)*torch.arange(0, self.num_samples+1,device='cuda')
        diffs=z_vals[1:]-z_vals[:-1]  #(num_samples,)
        loc=torch.rand((Nv,self.num_samples,),device='cuda')
        
        z_vals=z_vals[:-1].unsqueeze(0)+loc*diffs.unsqueeze(0)  #(Nv,num_samples,)
        
        ori=rays.origins.unsqueeze(1)  #(N,1,3)
        ray_dir=rays.directions
        ray_dir=ray_dir.unsqueeze(1)   #(N,1,3)
        z_vals=z_vals.unsqueeze(-1)  #(N,num_samples,1)
        ray_dir=ray_dir.broadcast_to((Nv,self.num_samples,3))
        
        sample_points=ori+z_vals*ray_dir
        
        return rays._replace(sample_points=sample_points, sample_lengths=z_vals*torch.ones_like(sample_points[...,-1:]))
    

