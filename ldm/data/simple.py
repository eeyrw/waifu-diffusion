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


def ResizeAndCrop(minSqureWidth, maxSquareWidth, img):
    minPixels = minSqureWidth*minSqureWidth
    maxPixels = maxSquareWidth*maxSquareWidth
    img_w, img_h = img.size
    targetPixels = random.randint(minPixels, maxPixels)
    imagePixels = img_w*img_h
    aspectRatio = img_h/img_w
    if not imagePixels > targetPixels:
        targetPixels = imagePixels

    resizedW = math.sqrt(targetPixels/aspectRatio)
    resizedH = resizedW*aspectRatio
    resizedW = int(resizedW)
    resizedW64x = resizedW-resizedW % 64
    resizedH = int(resizedH)
    resizedH64x = resizedH-resizedH % 64
    rsz = transforms.Resize(min(resizedW64x, resizedH64x),
                            interpolation=transforms.InterpolationMode.LANCZOS)
    crp = transforms.RandomCrop((resizedH64x, resizedW64x), pad_if_needed=True)
    img = rsz(img)
    img = crp(img)
    return img


class ImageInfoDs(Dataset):
    def __init__(self, root_dir, image_transforms=None, ucg=0.1, mode='train',
                 val_split=10, imageSquareWidthRange=(512, 768)) -> None:
        self.root_dir = Path(root_dir)
        imageInfoJsonPath = os.path.join(self.root_dir, 'ImageInfo.json')
        with open(imageInfoJsonPath, "r") as f:
            self.imageInfoList = json.load(f)

        if mode == 'train':
            self.imageInfoList = self.imageInfoList[val_split:-1]
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
        self.imageSquareWidthRange = imageSquareWidthRange

        # assert all(['full/' + str(x.name) in self.captions for x in self.paths])

    def __len__(self):
        return len(self.imageInfoList)

    def __getitem__(self, index):
        imageInfo = self.imageInfoList[index]
        imagePath = os.path.join(self.root_dir, imageInfo['IMG'])
        print(imagePath)
        im = Image.open(imagePath)
        im = self.process_im(im)
        captions = imageInfo['CAP']
        if captions is None or random.random() < self.ucg:
            caption = ""
        else:
            caption = random.choice(captions)
        return {"image": im, "caption": caption}

    def process_im(self, im):
        im = im.convert("RGB")
        im = ResizeAndCrop(
            self.imageSquareWidthRange[0], self.imageSquareWidthRange[1], im)
        if self.tform:
            im = self.tform(im)
        im = np.array(im).astype(np.uint8)
        return (im / 127.5 - 1.0).astype(np.float32)


def example():

    ds = ImageInfoDs(
        root_dir='/root/autodl-tmp/FinalDsWds',
        mode='train',
        val_split=200,
        imageSquareWidthRange=(512, 1280)
    )

    for i, it in enumerate(ds):
        image = ((it['image'] + 1) * 127.5).astype(np.uint8)
        image = Image.fromarray(image)
        image.save('./temp/example_%d.webp'%i)
        print(i)


if __name__ == '__main__':
    example()
