import io
import json
import re
import torch
import pytorch_lightning as pl
import torchvision
import webdataset as wds
import os
import numpy as np
import PIL
from PIL import Image, ImageOps
import random

PIL.Image.MAX_IMAGE_PIXELS = 933120000


def resize_image(image: Image, max_size=(768, 768)):
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


class ImageProcessor(object):
    def __init__(self, transforms, max_size, resize, image_key, ucg=0.1):
        self.transforms = transforms
        self.max_size = max_size
        self.resize = resize
        self.ucg = ucg
        self.image_key = image_key

    def __call__(self, sample):
        # preprocess caption
        imageInfo = json.loads(sample['json'])
        cap = imageInfo['CAP']
        if isinstance(cap,str):
            selectCap = cap
        elif isinstance(cap,list):
            selectCap = random.choice(imageInfo['CAP'])
        else:
            selectCap = ""
        if random.random() < self.ucg:
            selectCap = ""

        processedSample = dict()
        processedSample['caption'] = selectCap

        # preprocess image
        image = sample[self.image_key]
        image = Image.open(io.BytesIO(image),formats=['WEBP'])
        if self.resize:
            image = resize_image(image, max_size=(
                self.max_size, self.max_size))
        image = self.transforms(image)
        image = np.array(image).astype(np.uint8)
        processedSample['image'] = (image / 127.5 - 1.0).astype(np.float32)
        return processedSample


def dict_collation_fn(samples, combine_tensors=False, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    # keys = set.intersection(*[set(sample.keys()) for sample in samples])
    # batched = {key: [] for key in keys}

    # for s in samples:
    #     [batched[key].append(s[key]) for key in batched]

    # result = {}
    # for key in batched:
    #     if isinstance(batched[key][0], (int, float)):
    #         if combine_scalars:
    #             result[key] = np.array(list(batched[key]))
    #     elif isinstance(batched[key][0], torch.Tensor):
    #         if combine_tensors:
    #             result[key] = torch.stack(list(batched[key]))
    #     elif isinstance(batched[key][0], np.ndarray):
    #         if combine_tensors:
    #             result[key] = np.array(list(batched[key]))
    #         else:
    #             result[key] = batched[key]
    #     else:
    #         result[key] = list(batched[key])
    # return result
    samples[0]['image'] = samples[0]['image'][None,...]
    samples[0]['caption'] = [samples[0]['caption']]
    return samples[0]


class SimpleWebDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, tar_base,
                 batch_size,
                 train=None, validation=None, test=None,
                 num_workers=4,
                 max_size=768, resize=False, flip_p=0, ucg=0.1,
                 image_key='image', **kwargs):
        super().__init__()
        print(f'Setting tar base to {tar_base}')
        self.tar_base = tar_base
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.validation = validation
        self.test = test
        self.max_size = max_size
        self.resize = resize
        self.flip_p = flip_p
        self.image_key = image_key
        self.ucg = ucg

    def make_loader(self, train=True):
        image_transforms = []
        image_transforms.extend(
            [torchvision.transforms.RandomHorizontalFlip(self.flip_p)],)
        image_transforms = torchvision.transforms.Compose(image_transforms)

        transform_dict = {}
        transform_dict.update({self.image_key: image_transforms})

        postprocess = ImageProcessor(
            transforms=image_transforms,
            max_size=self.max_size,
            resize=self.resize,
            image_key=self.image_key,
            ucg=self.ucg)

        tars = os.path.join(self.tar_base)

        dset = wds.WebDataset(
            tars,
            handler=wds.warn_and_continue).repeat().shuffle(1.0)
        print(f'Loading webdataset with {len(dset.pipeline[0].urls)} shards.')
        dset = (dset
                .select(self.filter_keys)
                )
        if postprocess is not None:
            dset = dset.map(postprocess)
        dset = (dset
                .batched(self.batch_size, partial=False,
                         collation_fn=dict_collation_fn)
                )

        loader = wds.WebLoader(dset, batch_size=None, shuffle=False,
                               num_workers=self.num_workers)

        return loader

    def filter_keys(self, x):
        return True

    def train_dataloader(self):
        return self.make_loader(self.train)

    def val_dataloader(self):
        return self.make_loader(train=False)

    def test_dataloader(self):
        return self.make_loader(train=False)


def example():
    from omegaconf import OmegaConf
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import IterableDataset
    from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler
    from pytorch_lightning.trainer.supporters import CombinedLoader, CycleIterator

    datamod = SimpleWebDataModuleFromConfig(
        '/root/autodl-tmp/FinalDsWds-{00000..00001}.tar',
        1,
        train=None, validation=None, test=None,
        num_workers=1,
        max_size=768, resize=True, flip_p=0.5,
        image_key='webp')

    dataloader = datamod.train_dataloader()

    for batch in dataloader:
        print(batch["image"].shape)
        print(batch['caption'])
        #image = ((batch["image"][0] + 1) * 127.5).numpy().astype(np.uint8)
        #image = Image.fromarray(image)
        #image.save('example.png')
        #break


if __name__ == '__main__':
    example()
