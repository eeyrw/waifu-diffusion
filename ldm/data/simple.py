from operator import mod
from typing import Dict
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange
from ldm.util import instantiate_from_config
import os
import random

class ImageInfoDs(Dataset):
    def __init__(self, root_dir,image_transforms,
    is_make_square=True,ucg=0.1,mode='train',
    val_split=10) -> None:
        self.root_dir = Path(root_dir)
        imageInfoJsonPath = os.path.join(self.root_dir,'ImageInfo.json')
        with open(imageInfoJsonPath, "r") as f:
            self.imageInfoList = json.load(f)

        if mode == 'train':
            self.imageInfoList = self.imageInfoList[val_split:-1]
        else:
            self.imageInfoList = self.imageInfoList[0:val_split]

        image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms
        self.is_make_square = is_make_square
        self.ucg = ucg

        # assert all(['full/' + str(x.name) in self.captions for x in self.paths])
            
    def _make_square(self, im, min_size=384, fill_color=(0, 0, 0, 0)):
        x, y = im.size
        size = max(min_size, x, y)
        new_im = Image.new('RGB', (size, size), fill_color)
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.imageInfoList)

    def __getitem__(self, index):
        imageInfo = self.imageInfoList[index]
        imagePath = os.path.join(self.root_dir,imageInfo['IMG'])
        im = Image.open(imagePath)
        im = self.process_im(im)
        caption = imageInfo['CAP']
        if caption is None or random.random() < self.ucg:
            caption = ""
        return {"image": im, "caption": caption}

    def process_im(self, im):
        im = im.convert("RGB")
        if self.is_make_square:
            im = self._make_square(im)
        im = self.tform(im)
        im = np.array(im).astype(np.uint8)
        return (im / 127.5 - 1.0).astype(np.float32)

