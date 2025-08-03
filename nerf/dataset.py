
from typing import Tuple,List
import os
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
from pytorch3d.renderer import PerspectiveCameras
ALL_DATASETS = ("lego")


DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data"
)



class ListDataset(Dataset):
    """
    A simple dataset made of a list of entries.
    """

    def __init__(self, entries: List) -> None:
        """
        Args:
            entries: The list of dataset entries.
        """
        self._entries = entries

    def __len__(
        self,
    ) -> int:
        return len(self._entries)

    def __getitem__(self, index):
        return self._entries[index]

def get_nerf_datasets(
    dataset_name: str,  
    image_size: Tuple[int, int],
    data_root: str = DEFAULT_DATA_ROOT):
    
    if dataset_name not in ALL_DATASETS:
        raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")

    print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")

    cameras_path = os.path.join(data_root, dataset_name + ".pth")
    image_path = cameras_path.replace(".pth", ".png")


    train_data = torch.load(cameras_path)
    n_cameras = train_data["cameras"]["R"].shape[0]

    _image_max_image_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None  
    images = torch.FloatTensor(np.array(Image.open(image_path))) / 255.0
    images = torch.stack(torch.chunk(images, n_cameras, dim=0))[..., :3]
    Image.MAX_IMAGE_PIXELS = _image_max_image_pixels

    scale_factors = [s_new / s for s, s_new in zip(images.shape[1:3], image_size)]

    if abs(scale_factors[0] - scale_factors[1]) > 1e-3:
        raise ValueError(
            "Non-isotropic scaling is not allowed. Consider changing the 'image_size' argument."
        )
    scale_factor = sum(scale_factors) * 0.5

    if scale_factor != 1.0:
        print(f"Rescaling dataset (factor={scale_factor})")
        images = torch.nn.functional.interpolate(
            images.permute(0, 3, 1, 2),
            size=tuple(image_size),
            mode="bilinear",
        ).permute(0, 2, 3, 1)

    cameras = [
        PerspectiveCameras(
            **{k: v[cami][None] for k, v in train_data["cameras"].items()}
        ).to("cpu")
        for cami in range(n_cameras)
    ]

    train_idx, val_idx, test_idx = train_data["split"]

    train_dataset, val_dataset, test_dataset = [
        ListDataset(
            [
                {"image": images[i], "camera": cameras[i], "camera_idx": int(i)}
                for i in idx
            ]
        )
        for idx in [train_idx, val_idx, test_idx]
    ]
    print('success')
    return train_dataset, val_dataset, test_dataset

if __name__ =='__main__':
    train_dataset, val_dataset, test_dataset=get_nerf_datasets('lego', image_size=(128,128))
    
    
    
    
    
    
    
    
    
    
    
    
    