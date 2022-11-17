from configparser import Interpolation
from operator import mod
from typing import Dict, final
import numpy as np
from omegaconf import DictConfig, ListConfig
from regex import F
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
import math


def ResizeAndCrop(img, resizeStrategy, buckets):
    minPixels = min(buckets)*min(buckets)
    maxPixels = max(buckets)*max(buckets)
    img_w, img_h = img.size
    imagePixels = img_w*img_h
    aspectRatio = img_h/img_w
    if resizeStrategy == 'random':
        targetPixels = random.randint(minPixels, maxPixels)
        if imagePixels < targetPixels:
            targetPixels = imagePixels
    elif resizeStrategy == 'maxSize':
        targetPixels = min(imagePixels,maxPixels)
    elif resizeStrategy == 'randomBuckets':
        targeWidth = random.choice(buckets)
        targetPixels = targeWidth*targeWidth

    resizedW = math.sqrt(targetPixels/aspectRatio)
    resizedH = resizedW*aspectRatio
    resizedW = int(resizedW)
    resizedW64x = resizedW-resizedW % 64
    resizedH = int(resizedH)
    resizedH64x = resizedH-resizedH % 64
    rsz = transforms.Resize(min(resizedW64x, resizedH64x),
                            interpolation=transforms.InterpolationMode.BICUBIC)
    crp = transforms.RandomCrop((resizedH64x, resizedW64x), pad_if_needed=True)
    img = rsz(img)
    img = crp(img)
    return img


class ImageInfoDs(Dataset):
    def __init__(self, root_dir, image_transforms=None, ucg=0.1, mode='train',
                 val_split=10,
                 imageSizeBuckets = (512, 768), 
                 resizeStrategy='random') -> None:
        self.root_dir = Path(root_dir)
        imageInfoJsonPath = os.path.join(self.root_dir, 'ImageInfo.json')
        with open(imageInfoJsonPath, "r") as f:
            self.imageInfoList = json.load(f)
            random.shuffle(self.imageInfoList)

        if mode == 'train': 
            self.imageInfoList = self.imageInfoList[val_split:]
        else:
            self.imageInfoList = self.imageInfoList[0:val_split]

        if image_transforms:
            image_transforms = [instantiate_from_config(
                tt) for tt in image_transforms]
            image_transforms = transforms.Compose(image_transforms)
            self.tform = image_transforms
        else:
            self.tform = None
        self.ucg = ucg
        self.imageSizeBuckets = imageSizeBuckets
        self.resizeStrategy = resizeStrategy

        # assert all(['full/' + str(x.name) in self.captions for x in self.paths])

    def __len__(self):
        return len(self.imageInfoList)

    def __getitem__(self, index):
        imageInfo = self.imageInfoList[index]
        imagePath = os.path.join(self.root_dir, imageInfo['IMG'])
        im = Image.open(imagePath)
        im = self.process_im(im)
        captions = imageInfo['CAP']
        if captions is None or random.random() < self.ucg:
            caption = ""
        else:
            if isinstance(captions,list):
                caption = random.choice(captions)
            elif isinstance(captions,str):
                caption = captions
        return {"image": im, "caption": caption}

    def process_im(self, im):
        im = im.convert("RGB")
        im = ResizeAndCrop(im,self.resizeStrategy,self.imageSizeBuckets)
        if self.tform:
            im = self.tform(im)
        im = np.array(im).astype(np.uint8)
        return (im / 127.5 - 1.0).astype(np.float32)


def example():

    ds = ImageInfoDs(
        root_dir='/root/autodl-tmp/FinalDs',
        mode='train',
        val_split=200,
        resizeStrategy='maxSize',
        imageSizeBuckets=(768, 1024)
    )

    for i, it in enumerate(ds):
        image = ((it['image'] + 1) * 127.5).astype(np.uint8)
        image = Image.fromarray(image)
        image.save('./temp/example_%d.webp'%i)
        print(i)


if __name__ == '__main__':
    example()
