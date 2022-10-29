import os
import numpy as np
import PIL
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from functools import partial
import copy

import glob

import random

PIL.Image.MAX_IMAGE_PIXELS = 933120000
import torchvision

import pytorch_lightning as pl
from pathlib import Path
import torch

import re
import json
import io

def resize_image(image: Image, max_size=(768,768)):
    image = ImageOps.contain(image, max_size, Image.Resampling.LANCZOS)
    # resize to integer multiple of 64
    w, h = image.size
    w, h = map(lambda x: x - x % 64, (w, h))

    ratio = w / h
    src_ratio = image.width / image.height

    src_w = w if ratio > src_ratio else image.width * h // image.height
    src_h = h if ratio <= src_ratio else image.height * w // image.width

    resized = image.resize((src_w, src_h), resample=Image.Resampling.LANCZOS)
    res = Image.new("RGB", (w, h))
    res.paste(resized, box=(w // 2 - src_w // 2, h // 2 - src_h // 2))

    return res

class CaptionProcessor(object):
    def __init__(self, transforms, max_size, resize, LR_size):
        self.transforms = transforms
        self.max_size = max_size
        self.resize = resize
        self.degradation_process = partial(TF.resize, size=LR_size, interpolation=TF.InterpolationMode.NEAREST)
    def _make_square(self, im, min_size=512, fill_color=(0, 0, 0, 0)):
        x, y = im.size
        size = max(min_size, x, y)
        new_im = Image.new('RGB', (size, size), fill_color)
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
        new_im = new_im.resize((512,512))
        return new_im    
    def __call__(self, imagePath):
        sample = dict()
        image = Image.open(imagePath)
        if self.resize:
            image =self._make_square(image,512) # resize_image(image, max_size=(self.max_size, self.max_size))
        image = self.transforms(image)
        image = np.array(image).astype(np.uint8)
        sample['image'] = (image / 127.5 - 1.0).astype(np.float32)

        return sample

class SimpleVAE(Dataset):
    def __init__(self,
                 root_dir='./danbooru-aesthetic',
                 size=256,
                 flip_p=0.5,
                 mode='train',
                 val_split=64,
                 downscale_f=8
                 ):
        super().__init__()
        print('Fetching data.')

        self.root_dir = Path(root_dir)
        imageInfoJsonPath = os.path.join(self.root_dir,'ImageInfo.json')
        with open(imageInfoJsonPath, "r") as f:
            self.imageInfoList = json.load(f)

        if mode == 'train':
            self.imageInfoList = self.imageInfoList[val_split:-1]
        else:
            self.imageInfoList = self.imageInfoList[0:val_split]


        self.size = size
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        image_transforms = []
        image_transforms.extend([torchvision.transforms.RandomHorizontalFlip(flip_p)],)
        image_transforms = torchvision.transforms.Compose(image_transforms)

        self.captionprocessor = CaptionProcessor(image_transforms, self.size, True, int(size / downscale_f))

    def random_sample(self):
        return self.__getitem__(random.randint(0, self.__len__() - 1))
    
    def sequential_sample(self, i):
        if i >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(i + 1)

    def skip_sample(self, i):
        return None

    def __len__(self):
        return len(self.imageInfoList)

    def __getitem__(self, i):
        return self.get_image(i)
    
    def get_image(self, i):
        image = {}
        try:
            image = self.captionprocessor(os.path.join(self.root_dir,self.imageInfoList[i]['IMG']))
        except Exception as e:
            print(f'Error with %s -- %s -- skipping %s'%(self.imageInfoList[i]['IMG'],e,i))
            return self.skip_sample(i)
        
        return image

if __name__ == "__main__":
    dataset = SimpleVAE('/root/autodl-tmp/FinalDs', size=512, mode='train')
    print(len(dataset))
    example = dataset[0]
    image = example['image']
    image = ((image + 1) * 127.5).astype(np.uint8)
    image = Image.fromarray(image)
    image.save('example.png')
"""
if __name__ == "__main__":
    dataset = LocalBase('./danbooru-aesthetic', size=512, crop=False, mode='val')
    print(dataset.__len__())
    example = dataset.__getitem__(0)
    print(dataset.hashes[0])
    print(example['caption'])
    image = example['image']
    image = ((image + 1) * 127.5).astype(np.uint8)
    image = Image.fromarray(image)
    image.save('example.png')
"""
"""
from tqdm import tqdm
if __name__ == "__main__":
    dataset = LocalDanbooruBase('./links', size=768)
    import time
    a = time.process_time()
    for i in range(8):
        example = dataset.get_image(i)
        image = example['image']
        image = ((image + 1) * 127.5).astype(np.uint8)
        image = Image.fromarray(image)
        image.save(f'example-{i}.png')
        print(example['caption'])
    print('time:', time.process_time()-a)
"""
