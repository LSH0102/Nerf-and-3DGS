
import torch
import torch.nn as nn
import implicit_func
import sampler
import renderer
import rays
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_nerf_datasets
import tqdm
import imageio

class NERF(nn.Module):
    def __init__(self):
        super().__init__()
        'Nerf的三个组件: 隐函数, 光线采样, 渲染器'
        
        self.f=implicit_func.NeuralRadianceField()
        
        self.sampler=sampler.UniformSampler(128,2.0,6.0)
        
        self.render=renderer.VolumeRenderer(chunk_size=32768)
        
    def forward(self,rays):
        '先采样获得采样点 然后和方向一起送入f得到color和depth 然后用render渲染得到img'
        
        
        return self.render(self.sampler,self.f,rays)
    
def render_img(model,cameras,image_shape,save=True):
    device='cuda'
    images=[]
    for ind, camera in enumerate(cameras):
        print(f'rendering image{ind}')
        
        torch.cuda.empty_cache()
        camera=camera.to(device)
        grid=rays.from_image_to_pixel(image_shape, camera)
        
        ray_bundle=rays.generate_rays(grid, image_shape, camera)
        
        out=model(ray_bundle)
        image=np.array(out['color'].view(image_shape[0],image_shape[1],3).detach().cpu())
        
        images.append(image)
        
        if save==True:
            plt.imsave(f'test_{ind}.png', image)
    return images

def col(batch):
    return batch

def train_nerf():
    model=NERF()
    model.to('cuda')
    opt=torch.optim.Adam(model.parameters(),lr=0.0005)
    
    train_dataset,val_dataset,_=get_nerf_datasets('lego', image_size=(128,128))
    
    train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=0,collate_fn=col)
    
    image_shape=(128,128)
    
    epochs=6
    
    batch_size=1024
    for i in range(0,epochs):
        t_range = tqdm.tqdm(enumerate(train_dataloader))

        for iteration, batch in t_range:
            image, camera, camera_idx = batch[0].values()
            image = image.cuda().unsqueeze(0)
            camera = camera.cuda()

            xy_grid = rays.from_image_to_pixel_random(
                batch_size, (128,128), camera
            )
            ray_bundle = rays.generate_rays(
                xy_grid, (128,128), camera
            )
            rgb_gt = rays.get_color_grid(image, xy_grid)

            out = model(ray_bundle)

            loss = torch.nn.MSELoss()(out['color'],rgb_gt)

            opt.zero_grad()
            loss.backward()
            opt.step()

            t_range.set_description(f'Epoch: {i:04d}, Loss: {loss:.06f}')
            t_range.refresh()

        if i==5:
            with torch.no_grad():
                images=render_img(model, cameras=rays.create_cameras(radius=4.0,n_poses=20,up=(0.0,0.0,1.0),focal_length=2.0),
                                  image_shape=image_shape)
                imageio.mimsave('results/res.gif', [np.uint8(im*255) for im in images], loop=0)
                
if __name__=='__main__':
    train_nerf()
                
            
            
        
    
    
    
    
























































        
        
        
        