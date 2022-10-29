from operator import mod
from typing import Dict, final
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

def GenImageSizeBuckets(h,w):
    # https://blog.novelai.net/novelai-improvements-on-stable-diffusion-e10d38db82ac
    # ● Set the width to 256.
    # ● While the width is less than or equal to 1024:
    # • Find the largest height such that height is less than or equal to 1024 and that width multiplied by height is less than or equal to 512 * 768.
    # • Add the resolution given by height and width as a bucket.
    # • Increase the width by 64.

    maxPixelNum =h*w
    width = 256
    bucketSet=set()
    bucketSet.add((min(h,w),min(h,w))) # Add default size
    while width<=1280:
        height = min(maxPixelNum//width//64*64,1280)
        bucketSet.add((width,height))
        bucketSet.add((height,width))
        width = width+64
    
    return list(bucketSet)


def ResizeAndCrop(buckets,img):
    img_w, img_h = img.size
    bucketRatios = np.array([w/h for w,h in buckets])
    targetRatios = np.repeat(img_w/img_h,len(bucketRatios))
    bucketIndex = np.argmin(np.abs(bucketRatios-targetRatios))
    final_w,final_h = buckets[bucketIndex]
    rsz = transforms.Resize(min(final_w,final_h))
    crp = transforms.RandomCrop((final_h,final_w),pad_if_needed=True)
    img = rsz(img)
    img = crp(img)
    return img
    


class ImageInfoDs(Dataset):
    def __init__(self, root_dir,image_transforms=None,
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

        if image_transforms:
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
            image_transforms = transforms.Compose(image_transforms)
            self.tform = image_transforms
        else:
            self.tform = None
        self.is_make_square = is_make_square
        self.ucg = ucg
        self.big_buckets = GenImageSizeBuckets(1280,1280)
        self.small_buckets = GenImageSizeBuckets(768,768)

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
        if random.random() < 0.5:
            im = ResizeAndCrop(self.small_buckets, im)
        else:
            im = ResizeAndCrop(self.big_buckets, im)
        if self.tform:
            im = self.tform(im)
        im = np.array(im).astype(np.uint8)
        return (im / 127.5 - 1.0).astype(np.float32)

