import torch
import torchvision
import os
import glob
import random
import tqdm
import itertools
import numpy as np
import json
import re
import shutil


from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageOps
from PIL.Image import Image as Img

from typing import Dict, List, Generator, Tuple

from pillow_heif import register_heif_opener
register_heif_opener()



class Validation():
    def __init__(self, is_skipped: bool, is_extended: bool) -> None:
        if is_skipped:
            self.validate = self.__no_op
            return print("Validation: Skipped")

        if is_extended:
            self.validate = self.__extended_validate
            return print("Validation: Extended")

        self.validate = self.__validate
        print("Validation: Standard")

    def __validate(self, fp: str) -> bool:
        try:
            Image.open(fp)
            return True
        except:
            print(f'WARNING: Image cannot be opened: {fp}')
            return False

    def __extended_validate(self, fp: str) -> bool:
        try:
            Image.open(fp).load()
            return True
        except (OSError) as error:
            if 'truncated' in str(error):
                print(f'WARNING: Image truncated: {error}')
                return False
            print(f'WARNING: Image cannot be opened: {error}')
            return False
        except:
            print(f'WARNING: Image cannot be opened: {error}')
            return False

    def __no_op(self, fp: str) -> bool:
        return True



class ImageStore:
    def __init__(self, args, data_dir: str) -> None:
        self.data_dir = data_dir
        if os.path.isdir(self.data_dir):
            imageInfoJsonPath = os.path.join(self.data_dir, 'ImageInfo.json')
        elif os.path.isfile(self.data_dir):
            imageInfoJsonPath = self.data_dir
            self.data_dir = os.path.dirname(imageInfoJsonPath)
        with open(imageInfoJsonPath, "r") as f:
            self.imageInfoList = json.load(f)
            # random.seed(a=42, version=2)
            # random.shuffle(self.imageInfoList)

        imageInfoListFiltered = []
        for imageInfo in self.imageInfoList:
            if min(imageInfo['W'],imageInfo['H'])>=args.resolution:
                imageInfoListFiltered.append(imageInfo)
        self.imageInfoList = imageInfoListFiltered
        self.image_files = [os.path.join(
                     self.data_dir, imageInfo['IMG']) for imageInfo in self.imageInfoList]
        self.validator = Validation(
            args.skip_validation,
            args.extended_validation
        ).validate

        self.image_files = [x for x in self.image_files if self.validator(x)]

    def __len__(self) -> int:
        return len(self.image_files)

    # get image by index
    def get_image(self, idx) -> Img:
        return Image.open(self.image_files[idx]).convert(mode='RGB')

    # gets caption by removing the extension from the filename and replacing it with .txt
    def get_caption(self, idx) -> str:
        qualityDescList = []
        isNegativeSample = False
        # if 'Q512' in self.imageInfoList[ref[0]].keys():
        #     Q = self.imageInfoList[ref[0]]['Q512']
        #     if Q>65:
        #         qualityDescList.append('high res,best quality')
        #     elif Q<50:
        #         qualityDescList.append('low res,low quality')
        #         isNegativeSample = True

        # if 'A' in self.imageInfoList[ref[0]].keys():
        #     A = self.imageInfoList[ref[0]]['A']
        #     if A>5.5:
        #         qualityDescList.append('masterpiece')
        #     elif A<3:
        #         qualityDescList.append('bad art')
        #         # isNegativeSample = True

        if 'CAP' in self.imageInfoList[idx].keys():
            captions = self.imageInfoList[idx]['CAP']
        else:
            captions = None
            #print(self.imageInfoList[ref[0]]['IMG'])
        if captions is None:
            caption = ""
        else:
            if isinstance(captions, list):
                caption = random.choice(captions)
            elif isinstance(captions, str):
                caption = captions
            if 'artist' in self.imageInfoList[idx].keys():
                caption = 'by artist '+ self.imageInfoList[idx]['artist'] + caption
            # if 'style' in self.imageInfoList[ref[0]].keys():
            #     caption = 'in style of '+ self.imageInfoList[ref[0]]['style'] + caption
            
            
        #caption = ','.join(qualityDescList)+',' + caption
        return caption



class AspectDataset(torch.utils.data.Dataset):
    def __init__(self, args, store: ImageStore, device: torch.device, ucg: float = 0.1):
        self.store = store
        self.device = device
        self.ucg = ucg
        self.args = args


        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(args.resolution),
            torchvision.transforms.RandomCrop(args.resolution),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.store)

    def __getitem__(self, idx):
        return_dict = {'pixel_values': None, 'input_text': None}

        image_file = self.store.get_image(idx)

        return_dict['pixel_values'] = self.transforms(image_file)
        if random.random() > self.ucg:
            caption_file = self.store.get_caption(idx)
        else:
            caption_file = ''

        return_dict['input_text'] = caption_file
        return return_dict

    def collate_fn(self, examples):
        pixel_values = torch.stack([example['pixel_values']
                                   for example in examples if example is not None])
        pixel_values.to(memory_format=torch.contiguous_format).float()
        input_texts = [example['input_text']
                                   for example in examples if example is not None]
        return {
            'pixel_values': pixel_values,
            'input_texts': input_texts,
        }


class FixedCropDataloader:
    def __init__(self, args, device, world_size, rank) -> None:
        self.store = ImageStore(args,args.train_data_dir)
        self.dataset = AspectDataset(
            args, self.store, device, ucg=args.ucg)
        self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        print(f'STORE_LEN: {len(self.store)}')
        self.train_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=self.sampler,
            num_workers=4,
            collate_fn=self.dataset.collate_fn
        )

if __name__ == "__main__":
    pass