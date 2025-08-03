

import torch 
import torch.nn as nn

class Positional_Enconding(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()
        self.embed_dim=embed_dim
        factors=torch.arange(0, self.embed_dim,device='cuda')
        self.factors=torch.pow(2, factors).unsqueeze(0)
        self.output_dim=self.embed_dim*6
        
    def forward(self, x:torch.Tensor):
        'x: (N,3)代表三维坐标或者向量方向'
        
        x=x.view(-1,1)
        x=self.factors*x
        sin=torch.sin(x)
        cos=torch.cos(x)
        
        emb=torch.stack((sin,cos),dim=1).reshape((-1,2*self.embed_dim))
        emb=emb.view(-1,6*self.embed_dim)
        return emb
    
class NeuralRadianceField(nn.Module):
    def __init__(self):
        super().__init__()
        self.xyz_embd=Positional_Enconding(6)
        self.dir_embd=Positional_Enconding(2)
        
        self.xyz_dim=self.xyz_embd.output_dim
        self.dir_dim=self.dir_embd.output_dim
        
        self.L1=torch.nn.Linear(self.xyz_dim+self.dir_dim, out_features=256)
        
        self.relu=torch.nn.ReLU()
        self.L2=torch.nn.Linear(in_features=256, out_features=512)
        self.L3=torch.nn.Linear(in_features=512, out_features=128)
        self.L4=torch.nn.Linear(in_features=128, out_features=4)
        self.sig=torch.nn.Sigmoid()
        
    def forward(self, x):
        'x是经过sampler采样过的光线 坐标为x.sample_points (N,n_per_ray,3) 方向为x.directions (N,3)'
        '注意全化为(N,n_per_ray,3)然后再进行positional_embedding'
        points=x.sample_points
        dirs=x.directions.unsqueeze(1)
        n_per_ray=points.shape[1]
        dirs=dirs.broadcast_to((dirs.shape[0],n_per_ray,3))
        points=points.view(-1,3)
        dirs=dirs.reshape((-1,3))
        points=self.xyz_embd(points)
        dirs=self.dir_embd(dirs)
        y=torch.concat([points,dirs],dim=-1)
        
        y=self.L1(y)
        y=self.relu(y)
        y=self.L2(y)
        y=self.relu(y)
        y=self.L3(y)
        y=self.relu(y)
        y=self.L4(y)
        color=y[...,:3]
        density=y[...,-1:]
        
        color=self.sig(color)
        density=self.relu(density)
        
        out=dict()
        out['color']=color
        out['density']=density
        return out
    
        
        
        