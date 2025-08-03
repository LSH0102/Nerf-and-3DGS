
import torch
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer import PerspectiveCameras,look_at_view_transform
import numpy as np


class Rays(object):
    def __init__(self,origin,directions,sample_points=None,sample_lengths=None):
        '一个从origin出发 向个点散射的光束的全体, 为方便运算,origins的shape=(N,3), 由N个相同的向量拼接而成'
        'directions:(N,3), 且每个方向向量的l2范数都是1,因此在传入参数前不要忘记用torch.functional里面的normalize'
        'sample_points: (N, num_points_per_ray, 3) 可以初始化为torh.zeros.(N,1,3)'
        'sample_lengths: (N,num_points_per_ray,1) 代表每个光线上第i个sample点到origin之间z坐标的差距 不是长度差距'
        
        self.device=origin.device   
        self.origins=origin
        self.directions=directions
        
        if sample_lengths==None:
            self.sample_lengths=torch.zeros((directions.shape[0],1,1),device=self.device)
        else:
            self.sample_lengths=sample_lengths
        if sample_points==None:
            self.sample_points=torch.zeros((directions.shape[0],1,3),device=self.device)
        else:
            self.sample_points=sample_points
            
    def _replace(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        
        return self
    
    def __getitem__(self,ind):
        return Rays(self.origins[ind],self.directions[ind],self.sample_points[ind],self.sample_lengths[ind])
        
def from_image_to_pixel(image_shape,camera:CamerasBase):
    '把(H,W)的图片缩放为[-1,1]的正方形网格'
    H,W=image_shape
    x=np.arange(0,H )
    y=np.arange(0,W )
    
    x=2*x/H-1.0
    y=2*y/W-1.0
    x=torch.Tensor(x).to('cuda')
    y=torch.Tensor(y).to('cuda')   
    
    grid=torch.stack(tuple(reversed(torch.meshgrid(y,x))),dim=-1).view(H*W,2)
    return -grid

def from_image_to_pixel_random(n_samples,image_shape,camera):
    '随机选取一些网格点'
    grid=from_image_to_pixel(image_shape, camera)
    rand=torch.randint(0, grid.shape[0], size=(n_samples,1),device='cuda')
    rand=rand.broadcast_to((n_samples,2))
    grid_rand=torch.gather(grid, dim=0, index=rand)
    return grid_rand.reshape((-1,2))[:n_samples]

def generate_rays(grid,image_shape,camera:CamerasBase):
    '生成光束 '
    
    
    ndc_coor=torch.concat([grid,torch.ones_like(grid[...,-1:])],dim=-1)  #(H*W,3) ndc空间坐标
    
    world_sapce_points=camera.unproject_points(ndc_coor,world_coordinates=True,from_ndc=True)
    
    origins=camera.get_camera_center() #(1,3)
    origins=origins.broadcast_to((grid.shape[0],3))
    
    ray_dirs=world_sapce_points-origins
    ray_dirs=torch.nn.functional.normalize(ray_dirs,dim=-1)
    
    return Rays(origins, directions=ray_dirs)

def get_color_grid(images,grid):
    '获得grid格点出image的图像颜色信息'
    N=images.shape[0]
    grid=-grid.view((N,-1,1,2))
    
    imgs=torch.nn.functional.grid_sample(images.permute(0,3,1,2), grid,align_corners=True,mode='bilinear')
    
    return imgs.permute(0,2,3,1).view(-1,images.shape[-1]) 

def create_cameras(radius, n_poses=20,up=(0.0,1.0,0.0),focal_length=1.0):
    cameras = []

    for theta in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:

        if np.abs(up[1]) > 0:
            eye = [np.cos(theta + np.pi / 2) * radius, 0, -np.sin(theta + np.pi / 2) * radius]
        else:
            eye = [np.cos(theta + np.pi / 2) * radius, np.sin(theta + np.pi / 2) * radius, 2.0]

        R, T = look_at_view_transform(
            eye=(eye,),
            at=([0.0, 0.0, 0.0],),
            up=(up,),
        )

        cameras.append(
            PerspectiveCameras(
                focal_length=torch.tensor([focal_length])[None],
                principal_point=torch.tensor([0.0, 0.0])[None],
                R=R,
                T=T,
            )
        )
    
    return cameras
    


    
