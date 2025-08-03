

import torch
import torch.nn as nn



class VolumeRenderer(nn.Module):
    def __init__(self,chunk_size):
        super().__init__()
        self.chunk_size=chunk_size
    
    def compute_weights(self,density,delta_t):
        cumed_delta=-1.0*torch.cumsum(density*delta_t, dim=1)
        Ts=torch.concat([torch.ones_like(cumed_delta[...,-1:,:]),torch.exp(cumed_delta)[...,:-1,:]],dim=1)
        
        alpha=1.0-torch.exp(-1.0*delta_t*density)
        weights=Ts*alpha
        return weights
    
    def compute_sum(self,weights,summands):
        return (weights*summands).sum(dim=1)
    
    def forward(self,sampler, f,rays):
        batch_size=rays.directions.shape[0]
        
        chunks=[]
        for i in range(0,batch_size,self.chunk_size):
            mini_ray=rays[i:i+self.chunk_size]
            
            sampled_pts=sampler(mini_ray)
            num_samples_per_ray=sampled_pts.sample_points.shape[1]
            
            out=f(sampled_pts)
            density=out['density']
            color=out['color']
            
            depth=sampled_pts.sample_lengths[...,0]
            delta_t=torch.concat([depth[...,1:]-depth[...,:-1],1e10*torch.ones_like(depth[...,:1])],dim=-1)[...,None]
            
            weights=self.compute_weights(density.view((-1,num_samples_per_ray,1)), delta_t.view((-1,num_samples_per_ray,1)))
            
            color=self.compute_sum(weights, summands=color.view((-1,num_samples_per_ray,3)))
            
            depth=self.compute_sum(weights, depth.view((-1,num_samples_per_ray,1)))
            depth=depth/depth.max()
            
            chunk_out={'color':color,'depth':depth,}
            chunks.append(chunk_out)
        
        out={k:torch.concat([chunk_out[k] for chunk_out in chunks]) for k in chunks[0].keys()}
        
        return out
    

            
            
            
            
            
            
            
            
            