import argparse
from math import ceil
import math
import platform
import torch
import torchvision
import os
import random
import tqdm
import itertools
import numpy as np
import json
import re
import shutil
from safetensors.torch import save_file

from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, GPTNeoModel, GPT2Tokenizer, AutoTokenizer, T5Tokenizer, T5EncoderModel
from PIL import Image, ImageOps
from PIL.Image import Image as Img

from typing import Dict, List, Generator, Tuple
from scipy.interpolate import interp1d
from pillow_heif import register_heif_opener
register_heif_opener()
# defaults should be good for everyone
# TODO: add custom VAE support. should be simple with diffusers


def bool_t(x): return x.lower() in ['true', 'yes', '1']


parser = argparse.ArgumentParser(description='Aspect Latent Dataset Maker')

parser.add_argument('--dataset', type=str, default=None, required=True,
                    help='The path to the dataset to use for finetuning.')
parser.add_argument('--num_buckets', type=int, default=16,
                    help='The number of buckets.')
parser.add_argument('--bucket_side_min', type=int, default=256,
                    help='The minimum side length of a bucket.')
parser.add_argument('--bucket_side_max', type=int, default=2048,
                    help='The maximum side length of a bucket.')
parser.add_argument('--bucket_mode', type=str, default='multiscale',
                    help='multiscale|maxfit')
parser.add_argument('--resolution', type=int, default=512,
                    help='Image resolution to train against. Lower res images will be scaled up to this resolution and higher res images will be scaled down.')
parser.add_argument('--clip_penultimate', type=bool_t, default='False',
                    help='Use penultimate CLIP layer for text embedding')
parser.add_argument('--output_bucket_info', type=bool_t,
                    default='False', help='Outputs bucket information and exits')
parser.add_argument('--resize', type=bool_t, default='False',
                    help="Resizes dataset's images to the appropriate bucket dimensions.")
parser.add_argument('--use_xformers', type=bool_t,
                    default='False', help='Use memory efficient attention')
parser.add_argument('--extended_validation', type=bool_t, default='False',
                    help='Perform extended validation of images to catch truncated or corrupt images.')
parser.add_argument('--no_migration', type=bool_t, default='False',
                    help='Do not perform migration of dataset while the `--resize` flag is active. Migration creates an adjacent folder to the dataset with <dataset_dirname>_cropped.')
parser.add_argument('--skip_validation', type=bool_t, default='False',
                    help='Skip validation of images, useful for speeding up loading of very large datasets that have already been validated.')
parser.add_argument('--extended_mode_chunks', type=int, default=0,
                    help='Enables extended mode for tokenization with given amount of maximum chunks. Values < 2 disable.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--model', type=str, default=None, required=True,
                    help='The name of the model to use for finetuning. Could be HuggingFace ID or a directory')
parser.add_argument('--model_cache_dir', type=str, default=None, required=True,
                    help='The name of the model cache directory')
parser.add_argument('--output_dir', type=str, default='./latents', required=True,
                    help='The name of the model cache directory')
parser.add_argument('--text_encoder', type=str, default='CLIP', required=True,
                    help='The text encoder used, can be CLIP,GPT1.3B,T5v11-L')

args = parser.parse_args()

def setup():
    if platform.system()=='Windows':
        pass # Do nothing because Windows is not able to run parallel
    elif platform.system()=='Linux':
        torch.distributed.init_process_group("nccl", init_method="env://")

def cleanup():
    torch.distributed.destroy_process_group()

def get_rank() -> int:
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def get_world_size() -> int:
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def _sort_by_ratio(bucket: tuple) -> float:
    return bucket[0] / bucket[1]


def _sort_by_area(bucket: tuple) -> float:
    return bucket[0] * bucket[1]


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


class Resize():
    def __init__(self, is_resizing: bool, is_not_migrating: bool) -> None:
        if not is_resizing:
            self.resize = self.__no_op
            return

        if not is_not_migrating:
            self.resize = self.__migration
            dataset_path = os.path.split(args.dataset)
            self.__directory = os.path.join(
                dataset_path[0],
                f'{dataset_path[1]}_cropped'
            )
            os.makedirs(self.__directory, exist_ok=True)
            return print(f"Resizing: Performing migration to '{self.__directory}'.")

        self.resize = self.__no_migration

    def __no_migration(self, image_path: str, w: int, h: int) -> Img:
        return ImageOps.fit(
            Image.open(image_path),
            (w, h),
            bleed=0.0,
            centering=(0.5, 0.5),
            method=Image.Resampling.LANCZOS
        ).convert(mode='RGB')

    def __migration(self, image_path: str, w: int, h: int) -> Img:
        filename = re.sub('\.[^/.]+$', '', os.path.split(image_path)[1])

        image = ImageOps.fit(
            Image.open(image_path),
            (w, h),
            bleed=0.0,
            centering=(0.5, 0.5),
            method=Image.Resampling.LANCZOS
        ).convert(mode='RGB')

        image.save(
            os.path.join(f'{self.__directory}', f'{filename}.jpg'),
            optimize=True
        )

        return image

    def __no_op(self, image_path: str, w: int, h: int) -> Img:
        return Image.open(image_path)


class ImageStore:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        imageInfoJsonPath = os.path.join(self.data_dir, 'ImageInfo.json')
        with open(imageInfoJsonPath, "r") as f:
            self.imageInfoList = json.load(f)
            random.seed(a=42, version=2)
            random.shuffle(self.imageInfoList)
        self.image_files = [os.path.join(
            self.data_dir, imageInfo['IMG']) for imageInfo in self.imageInfoList]
        self.validator = Validation(
            args.skip_validation,
            args.extended_validation
        ).validate

        self.resizer = Resize(args.resize, args.no_migration).resize

        self.image_files = [x for x in self.image_files if self.validator(x)]

    def __len__(self) -> int:
        return len(self.image_files)

    # iterator returns images as PIL images and their index in the store
    def entries_iterator(self) -> Generator[Tuple[Img, int], None, None]:
        for f in range(len(self)):
            yield Image.open(self.image_files[f]), f

    # get image by index
    def get_image(self, ref: Tuple[int, int, int]) -> Img:
        return self.resizer(
            self.image_files[ref[0]],
            ref[1],
            ref[2]
        )

    # gets caption by removing the extension from the filename and replacing it with .txt
    def get_caption(self, ref: Tuple[int, int, int]) -> str:
        if 'CAP' in self.imageInfoList[ref[0]].keys():
            captions = self.imageInfoList[ref[0]]['CAP']
        else:
            captions = None
            print(self.imageInfoList[ref[0]]['IMG'])
        if captions is None:
            caption = ""
        else:
            if isinstance(captions, list):
                caption = random.choice(captions)
            elif isinstance(captions, str):
                caption = captions
            if 'artist' in self.imageInfoList[ref[0]].keys():
                caption = 'by artist '+ self.imageInfoList[ref[0]]['artist'] + caption
            if 'style' in self.imageInfoList[ref[0]].keys():
                caption = 'in style of '+ self.imageInfoList[ref[0]]['style'] + caption
        return caption

# ====================================== #
# Bucketing code stolen from hasuwoof:   #
# https://github.com/hasuwoof/huskystack #
# ====================================== #


class AspectBucket:
    def __init__(self, store: ImageStore,
                 num_buckets: int,
                 batch_size: int,
                 bucket_side_min: int = 256,
                 bucket_side_max: int = 768,
                 bucket_side_increment: int = 64,
                 bucket_mode: str = 'multiscale',
                 max_image_area: int = 512 * 768,
                 max_ratio: float = 2):

        self.requested_bucket_count = num_buckets
        self.bucket_length_min = bucket_side_min
        self.bucket_length_max = bucket_side_max
        self.bucket_increment = bucket_side_increment
        self.bucket_mode = bucket_mode
        self.max_image_area = max_image_area
        self.batch_size = batch_size
        self.total_dropped = 0

        if max_ratio <= 0:
            self.max_ratio = float('inf')
        else:
            self.max_ratio = max_ratio

        self.store = store
        self.buckets = []
        self._bucket_ratios = []
        self._bucket_interp = None
        self.bucket_data: Dict[tuple, List[int]] = dict()
        self.init_buckets()
        self.fill_buckets()

    def get_buckets(self,mode):
        if mode == 'maxfit':
            # https://blog.novelai.net/novelai-improvements-on-stable-diffusion-e10d38db82ac
            # ● Set the width to 256.
            # ● While the width is less than or equal to 1024:
            # • Find the largest height such that height is less than or equal to 1024 and that width multiplied by height is less than or equal to 512 * 768.
            # • Add the resolution given by height and width as a bucket.
            # • Increase the width by 64.

            maxPixelNum =self.max_image_area
            width = self.bucket_length_min
            bucketSet=set()
            #bucketSet.add((min(h,w),min(h,w))) # Add default size
            while width<=self.bucket_length_max:
                height = min(maxPixelNum//width//64*64,self.bucket_length_max)
                #if  max(width,height) / min(width,height) <= self.max_ratio:
                bucketSet.add((width,height))
                bucketSet.add((height,width))
                width = width+64
            possible_buckets = list(bucketSet)
        elif mode == 'multiscale':
            possible_lengths = list(
                range(self.bucket_length_min, self.bucket_length_max + 1, self.bucket_increment))
            possible_buckets = list((w, h) for w, h in itertools.product(possible_lengths, possible_lengths)
                                    if w >= h and w * h <= self.max_image_area and w / h <= self.max_ratio)
        return possible_buckets

    def init_buckets(self):

        possible_buckets = self.get_buckets(self.bucket_mode)

        buckets_by_ratio = {}

        # group the buckets by their aspect ratios
        for bucket in possible_buckets:
            w, h = bucket
            # use precision to avoid spooky floats messing up your day
            ratio = '{:.4e}'.format(w / h)

            if ratio not in buckets_by_ratio:
                group = set()
                buckets_by_ratio[ratio] = group
            else:
                group = buckets_by_ratio[ratio]

            group.add(bucket)

        # now we take the list of buckets we generated and pick the largest by area for each (the first sorted)
        # then we put all of those in a list, sorted by the aspect ratio
        # the square bucket (LxL) will be the first
        unique_ratio_buckets = sorted([sorted(buckets, key=_sort_by_area)[-1]
                                       for buckets in buckets_by_ratio.values()], key=_sort_by_ratio)

        # how many buckets to create for each side of the distribution
        bucket_count_each = int(
            np.clip((self.requested_bucket_count + 1) / 2, 1, len(unique_ratio_buckets)))

        # we know that the requested_bucket_count must be an odd number, so the indices we calculate
        # will include the square bucket and some linearly spaced buckets along the distribution
        indices = {
            *np.linspace(0, len(unique_ratio_buckets) - 1, bucket_count_each, dtype=int)}

        # make the buckets, make sure they are unique (to remove the duplicated square bucket), and sort them by ratio
        # here we add the portrait buckets by reversing the dimensions of the landscape buckets we generated above
        buckets = sorted({*(unique_ratio_buckets[i] for i in indices),
                          *(tuple(reversed(unique_ratio_buckets[i])) for i in indices)}, key=_sort_by_ratio)

        self.buckets = buckets

        # cache the bucket ratios and the interpolator that will be used for calculating the best bucket later
        # the interpolator makes a 1d piecewise interpolation where the input (x-axis) is the bucket ratio,
        # and the output is the bucket index in the self.buckets array
        # to find the best fit we can just round that number to get the index
        self._bucket_ratios = [w / h for w, h in buckets]
        self._bucket_interp = interp1d(self._bucket_ratios, list(range(len(buckets))), assume_sorted=True,
                                       fill_value=None)

        for b in buckets:
            self.bucket_data[b] = []

    def get_batch_count(self):
        return sum(ceil(len(b) / self.batch_size) for b in self.bucket_data.values())

    def get_bucket_info(self):
        return json.dumps({"buckets": self.buckets, "bucket_ratios": self._bucket_ratios})

    def get_batch_iterator(self) -> Generator[Tuple[Tuple[int, int, int]], None, None]:
        """
        Generator that provides batches where the images in a batch fall on the same bucket

        Each element generated will be:
            (index, w, h)

        where each image is an index into the dataset
        :return:
        """
        for b, values in self.bucket_data.items():
            batchList = [values[i:i+self.batch_size]
                         for i in range(0, len(values), self.batch_size)]
            for batch in batchList:
                yield [(idx, *b) for idx in batch]
        # max_bucket_len = max(len(b) for b in self.bucket_data.values())
        # index_schedule = list(range(max_bucket_len))
        # random.shuffle(index_schedule)

        # bucket_len_table = {
        #     b: len(self.bucket_data[b]) for b in self.buckets
        # }

        # bucket_schedule = []
        # for i, b in enumerate(self.buckets):
        #     bucket_schedule.extend([i] * (bucket_len_table[b] // self.batch_size))

        # bucket_pos = {
        #     b: 0 for b in self.buckets
        # }

        # total_generated_by_bucket = {
        #     b: 0 for b in self.buckets
        # }

        # for bucket_index in bucket_schedule:
        #     b = self.buckets[bucket_index]
        #     i = bucket_pos[b]
        #     bucket_len = bucket_len_table[b]

        #     batch = []
        #     while len(batch) != self.batch_size:
        #         # advance in the schedule until we find an index that is contained in the bucket
        #         k = index_schedule[i]
        #         if k < bucket_len:
        #             entry = self.bucket_data[b][k]
        #             batch.append(entry)

        #         i += 1

        #     total_generated_by_bucket[b] += self.batch_size
        #     bucket_pos[b] = i
        #     yield [(idx, *b) for idx in batch]

    def fill_buckets(self):
        entries = self.store.entries_iterator()
        total_dropped = 0

        for entry, index in tqdm.tqdm(entries, total=len(self.store)):
            if not self._process_entry(entry, index):
                total_dropped += 1

        # for b, values in self.bucket_data.items():
        #     # make sure the buckets have an exact number of elements for the batch
        #     to_drop = len(values) % self.batch_size
        #     self.bucket_data[b] = list(values[:len(values) - to_drop])
        #     total_dropped += to_drop

        self.total_dropped = total_dropped

    def _process_entry(self, entry: Image.Image, index: int) -> bool:
        aspect = entry.width / entry.height

        if aspect > self.max_ratio or (1 / aspect) > self.max_ratio:
            return False

        best_bucket = self._bucket_interp(aspect)

        if best_bucket is None:
            return False

        bucket = self.buckets[round(float(best_bucket))]

        self.bucket_data[bucket].append(index)

        del entry

        return True


class AspectBucketSampler(torch.utils.data.Sampler):
    def __init__(self, bucket: AspectBucket, num_replicas: int = 1, rank: int = 0):
        super().__init__(None)
        self.bucket = bucket
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        # subsample the bucket to only include the elements that are assigned to this rank
        indices = self.bucket.get_batch_iterator()
        indices = list(indices)[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        return int(math.ceil(self.bucket.get_batch_count() / self.num_replicas))


class LatentDatasetGenerator(torch.utils.data.Dataset):
    def __init__(self, store: ImageStore):
        self.store = store

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.store)

    def __getitem__(self, item: Tuple[int, int, int]):
        return_dict = {'pixel_values': None, 'input_ids': None}

        image_file = self.store.get_image(item)

        return_dict['pixel_values'] = self.transforms(image_file)
        caption_file = self.store.get_caption(item)

        return_dict['input_ids'] = caption_file
        return_dict['raw_item_index'] = item
        return return_dict

    def collate_fn(self, examples):
        pixel_values = torch.stack([example['pixel_values']
                                   for example in examples if example is not None])
        pixel_values.to(memory_format=torch.contiguous_format).float()

        return {
            'pixel_values': pixel_values,
            'input_ids': [example['input_ids'] for example in examples if example is not None],
            'raw_item_index': [example['raw_item_index'] for example in examples if example is not None]
        }

class TextEncoderGen:
    def __init__(self,args) -> None:
        self.textEncoderName = args.text_encoder
        self.args = args
        if self.textEncoderName == 'CLIP':
            self.tokenizer = CLIPTokenizer.from_pretrained(
                args.model, subfolder='tokenizer',cache_dir=args.model_cache_dir)
            self.text_encoder = CLIPTextModel.from_pretrained(
                args.model, subfolder='text_encoder',cache_dir=args.model_cache_dir)
        elif self.textEncoderName == 'GPT1.3B':
            self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B",cache_dir = args.model_cache_dir)
            self.text_encoder = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-1.3B",cache_dir = args.model_cache_dir)
        elif self.textEncoderName == 'T5v11-L':
            self.tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-large",cache_dir = args.model_cache_dir)
            self.text_encoder = T5EncoderModel.from_pretrained("google/t5-v1_1-large",cache_dir = args.model_cache_dir)
        else:
            raise RuntimeError('Unknown text encoder: %s'%args.text_encoder)     
    def inference(self,examples):
        if self.textEncoderName == 'CLIP':
            return self.textEncoderInferenceCLIP(examples)
        elif self.textEncoderName == 'GPT1.3B':
            return self.textEncoderInferenceGPT(examples)
        elif self.textEncoderName == 'T5v11-L':
            return self.textEncoderInferenceT5v11(examples)

    def textEncoderInferenceT5v11(self, examples):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        max_length = 200 #self.tokenizer.model_max_length - 2
        input_ids = []
        for example in examples['input_ids']:
            if example is not None:
                if example!='':
                    input_ids.append(self.tokenizer([example],truncation=True, return_length=True, return_overflowing_tokens=False, add_special_tokens=False, max_length=max_length).input_ids)
                else:
                    input_ids.append(self.tokenizer([example],padding='max_length',truncation=True, return_length=True, return_overflowing_tokens=False, add_special_tokens=False, max_length=50).input_ids)

        if self.args.clip_penultimate:
            input_ids = [self.text_encoder.text_model.final_layer_norm(self.text_encoder(torch.asarray(input_id).to(
                self.text_encoder.device), output_hidden_states=True)['hidden_states'][-2])[0] for input_id in input_ids]
        else:
            input_ids = [self.text_encoder(torch.asarray(input_id).to(
                self.text_encoder.device), output_hidden_states=True).last_hidden_state[0] for input_id in input_ids]

        # input_ids = torch.stack(tuple(input_ids))
        return input_ids

    def textEncoderInferenceGPT(self, examples):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        max_length = 200 #self.tokenizer.model_max_length - 2
        input_ids = []
        for example in examples['input_ids']:
            if example is not None:
                if example!='':
                    input_ids.append(self.tokenizer([example],truncation=True, return_length=True, return_overflowing_tokens=False, add_special_tokens=False, max_length=max_length).input_ids)
                else:
                    input_ids.append(self.tokenizer([example],padding='max_length',truncation=True, return_length=True, return_overflowing_tokens=False, add_special_tokens=False, max_length=50).input_ids)

        if self.args.clip_penultimate:
            input_ids = [self.text_encoder.text_model.final_layer_norm(self.text_encoder(torch.asarray(input_id).to(
                self.text_encoder.device), output_hidden_states=True)['hidden_states'][-2])[0] for input_id in input_ids]
        else:
            input_ids = [self.text_encoder(torch.asarray(input_id).to(
                self.text_encoder.device), output_hidden_states=True).last_hidden_state[0] for input_id in input_ids]

        # input_ids = torch.stack(tuple(input_ids))
        return input_ids

    def textEncoderInferenceCLIP(self, examples):
        if self.args.extended_mode_chunks < 2:
            max_length = self.tokenizer.model_max_length - 2
            input_ids = [self.tokenizer([example], truncation=True, return_length=True, return_overflowing_tokens=False,
                                padding='max_length', add_special_tokens=False, max_length=max_length).input_ids for example in examples['input_ids'] if example is not None]
        else:
            max_length = self.tokenizer.model_max_length
            max_chunks = self.args.extended_mode_chunks
            input_ids = [self.tokenizer([example], truncation=True, return_length=True, return_overflowing_tokens=False, padding='max_length',
                                add_special_tokens=False, max_length=(max_length * max_chunks) - (max_chunks * 2)).input_ids[0] for example in examples['input_ids'] if example is not None]
        if self.args.extended_mode_chunks < 2:
            for i, x in enumerate(input_ids):
                for j, y in enumerate(x):
                    input_ids[i][j] = [self.tokenizer.bos_token_id, *y, *np.full(
                        (self.tokenizer.model_max_length - len(y) - 1), self.tokenizer.eos_token_id)]

            if self.args.clip_penultimate:
                input_ids = [self.text_encoder.text_model.final_layer_norm(self.text_encoder(torch.asarray(input_id).to(
                    self.text_encoder.device), output_hidden_states=True)['hidden_states'][-2])[0] for input_id in input_ids]
            else:
                input_ids = [self.text_encoder(torch.asarray(input_id).to(
                    self.text_encoder.device), output_hidden_states=True).last_hidden_state[0] for input_id in input_ids]
        else:
            max_standard_tokens = max_length - 2
            max_chunks = args.extended_mode_chunks
            max_len = np.ceil(max(len(x) for x in input_ids) /
                            max_standard_tokens).astype(int).item() * max_standard_tokens
            if max_len > max_standard_tokens:
                z = None
                for i, x in enumerate(input_ids):
                    if len(x) < max_len:
                        input_ids[i] = [
                            *x, *np.full((max_len - len(x)), self.tokenizer.eos_token_id)]
                batch_t = torch.tensor(input_ids)
                chunks = [batch_t[:, i:i + max_standard_tokens]
                        for i in range(0, max_len, max_standard_tokens)]
                for chunk in chunks:
                    chunk = torch.cat((torch.full((chunk.shape[0], 1), self.tokenizer.bos_token_id), chunk, torch.full(
                        (chunk.shape[0], 1), self.tokenizer.eos_token_id)), 1)
                    if z is None:
                        if self.args.clip_penultimate:
                            z = self.text_encoder.text_model.final_layer_norm(self.text_encoder(
                                chunk.to(self.text_encoder.device), output_hidden_states=True)['hidden_states'][-2])
                        else:
                            z = self.text_encoder(
                                chunk.to(self.text_encoder.device), output_hidden_states=True).last_hidden_state
                    else:
                        if self.args.clip_penultimate:
                            z = torch.cat((z, self.text_encoder.text_model.final_layer_norm(self.text_encoder(
                                chunk.to(self.text_encoder.device), output_hidden_states=True)['hidden_states'][-2])), dim=-2)
                        else:
                            z = torch.cat((z, self.text_encoder(
                                chunk.to(self.text_encoder.device), output_hidden_states=True).last_hidden_state), dim=-2)
                input_ids = z
            else:
                for i, x in enumerate(input_ids):
                    input_ids[i] = [self.tokenizer.bos_token_id, *x, *np.full(
                        (self.tokenizer.model_max_length - len(x) - 1), self.tokenizer.eos_token_id)]
                if self.args.clip_penultimate:
                    input_ids = self.text_encoder.text_model.final_layer_norm(self.text_encoder(torch.asarray(
                        input_ids).to(self.text_encoder.device), output_hidden_states=True)['hidden_states'][-2])
                else:
                    input_ids = self.text_encoder(torch.asarray(input_ids).to(
                        self.text_encoder.device), output_hidden_states=True).last_hidden_state
        input_ids = torch.stack(tuple(input_ids))
        return input_ids


def VAEEncodeToLatent(vae, x):
    h = vae.encoder(x)
    moments = vae.quant_conv(h)
    return moments


if __name__ == "__main__":
    setup()
    rank = get_rank()
    world_size = get_world_size()
    torch.cuda.set_device(rank)
    # load dataset
    store = ImageStore(args.dataset)
    dataset = LatentDatasetGenerator(store)
    bucket = AspectBucket(store, args.num_buckets, args.batch_size, args.bucket_side_min,
                          args.bucket_side_max, 64, args.bucket_mode, args.resolution * args.resolution, 2.0)
    sampler = AspectBucketSampler(
        bucket=bucket, num_replicas=world_size, rank=rank)

    print(f'STORE_LEN: {len(store)}')

    if args.output_bucket_info:
        print(bucket.get_bucket_info())
        print('Num of drop samples: %d' % bucket.total_dropped)

    if rank==0:
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)    
        for bucketItem in bucket.buckets:
            sub_dir = '%dx%d'%(bucketItem[0],bucketItem[1])
            full_dir = os.path.join(args.output_dir,sub_dir)
            if not os.path.exists(full_dir):
                os.mkdir(full_dir)

    
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )

    device = torch.device('cuda')

    # if args.text_encoder == 'CLIP':
    #     tokenizer = CLIPTokenizer.from_pretrained(
    #         args.model, subfolder='tokenizer',cache_dir=args.model_cache_dir)
    #     text_encoder = CLIPTextModel.from_pretrained(
    #         args.model, subfolder='text_encoder',cache_dir=args.model_cache_dir)
    # elif args.text_encoder == 'GPT1.3B':
    #     tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B",cache_dir = args.model_cache_dir)
    #     text_encoder = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-1.3B",cache_dir = args.model_cache_dir)

    textEncoderGen = TextEncoderGen(args)

   
    vae = AutoencoderKL.from_pretrained(args.model, subfolder='vae',cache_dir=args.model_cache_dir)
    vae.requires_grad_(False)
    textEncoderGen.text_encoder.requires_grad_(False)
    vae = vae.to(device, dtype=torch.float32)
    textEncoderGen.text_encoder.to(device, dtype=torch.float16)
    toggle = True
    totalSamples = 0
    totalSamplesList = []
    for samples in tqdm.tqdm(train_dataloader):
        if toggle:
            latents = VAEEncodeToLatent(vae, samples['pixel_values'].to(
                device, dtype=torch.float32))

            textIds = textEncoderGen.inference(samples)
            toggle = False
        else:
            textIds =textEncoderGen.inference(samples)
            latents = VAEEncodeToLatent(vae, samples['pixel_values'].to(
                device, dtype=torch.float32))            
            toggle = True
        for idx,latent,textEmb in zip(samples['raw_item_index'],latents,textIds):
            tensors = {
                "imgLatent": latent.to(torch.float16),
                "txtEmb": textEmb.to(torch.float16)
            }
            fileName = '%dx%d/%d.safetensors'%(idx[1],idx[2],idx[0])
            save_file(tensors, os.path.join(args.output_dir,fileName))
        totalSamples+=latents.shape[0]

    if torch.distributed.is_initialized():
        totalSamplesList.append(torch.tensor(totalSamples,device = device))
        torch.distributed.all_reduce_multigpu(totalSamplesList,op=torch.distributed.ReduceOp.SUM)
        print('Rank %d: %d'%(get_rank(),totalSamples))
        if rank == 0:
            totalSamples = int(totalSamplesList[0].cpu())
            print('Total %d'%totalSamples)
    else:
        print('Total %d'% totalSamples)

    if rank==0: 
        metaJsonPath = os.path.join(args.output_dir,'LatentInfo.json')
        latentDict = {'%dx%d'%(k[0],k[1]):v for k,v in bucket.bucket_data.items()}

        unconditionalTxtEmb =textEncoderGen.inference({'input_ids':['']})[0]
        tensors = {
                "unconditionalTxtEmb": unconditionalTxtEmb.to(torch.float16)
            }
        save_file(tensors, os.path.join(args.output_dir,'UnconditionalTxtEmb.safetensors'))
        with open(metaJsonPath,'w',encoding='utf8') as f:
            json.dump(
                {
                    'TextEncoder':args.text_encoder,
                    'NumSamples':totalSamples,
                    'LatentDict':latentDict,
                    'Resolution':args.resolution,
                    'Model':args.model,
                },
                f)