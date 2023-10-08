from PIL import Image
import pathlib
from datasets import center_crop
import numpy as np
import einops
from torchvision.utils import make_grid, save_image
import os
import torch

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

file_path = '/home/data2/xiangyu/InstructTuning/Data/test_unidiffuser/GT'
path = pathlib.Path(file_path)
files = sorted([file for ext in IMAGE_EXTENSIONS
                for file in path.glob('*.{}'.format(ext))])
idx=0
for file in files:
    image = Image.open(file).convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = center_crop(512, 512, image).astype(np.float32)
    image = (image / 127.5 - 1.0).astype(np.float32)
    image = einops.rearrange(image, 'h w c -> c h w')
    image = torch.as_tensor(image)
    image = image.unsqueeze(0)
    save_path = os.path.join('/home/data2/xiangyu/InstructTuning/Data/test_unidiffuser/gt_512', f'{idx}_gt.png')
    save_image(image, save_path)
    idx=idx+1


