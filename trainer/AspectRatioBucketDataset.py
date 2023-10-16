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
from scipy.interpolate import interp1d

from pillow_heif import register_heif_opener
register_heif_opener()

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
    def __init__(self, args, is_resizing: bool, is_not_migrating: bool) -> None:
        if not is_resizing:
            self.resize = self.__no_op
            return

        if not is_not_migrating:
            self.resize = self.__migration
            dataset_path = os.path.split(args.train_data_dir)
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
            method=Image.Resampling.BICUBIC
        ).convert(mode='RGB')

    def __migration(self, image_path: str, w: int, h: int) -> Img:
        filename = re.sub('\.[^/.]+$', '', os.path.split(image_path)[1])

        image = ImageOps.fit(
            Image.open(image_path),
            (w, h),
            bleed=0.0,
            centering=(0.5, 0.5),
            method=Image.Resampling.BICUBIC
        ).convert(mode='RGB')

        image.save(
            os.path.join(f'{self.__directory}', f'{filename}.jpg'),
            optimize=True
        )

        return image

    def __no_op(self, image_path: str, w: int, h: int) -> Img:
        return Image.open(image_path)


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
        self.image_files = [os.path.join(
            self.data_dir, imageInfo['IMG']) for imageInfo in self.imageInfoList]
        self.validator = Validation(
            args.skip_validation,
            args.extended_validation
        ).validate

        self.resizer = Resize(args, args.resize, args.no_migration).resize

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

        if 'CAP' in self.imageInfoList[ref[0]].keys():
            captions = self.imageInfoList[ref[0]]['CAP']
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
            if 'artist' in self.imageInfoList[ref[0]].keys():
                caption = 'by artist '+ self.imageInfoList[ref[0]]['artist'] + caption
            # if 'style' in self.imageInfoList[ref[0]].keys():
            #     caption = 'in style of '+ self.imageInfoList[ref[0]]['style'] + caption
            
            
        #caption = ','.join(qualityDescList)+',' + caption
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
                 max_image_area: int = 512 * 768,
                 max_ratio: float = 2):

        self.requested_bucket_count = num_buckets
        self.bucket_length_min = bucket_side_min
        self.bucket_length_max = bucket_side_max
        self.bucket_increment = bucket_side_increment
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

    def init_buckets(self):
        possible_lengths = list(
            range(self.bucket_length_min, self.bucket_length_max + 1, self.bucket_increment))
        possible_buckets = list((w, h) for w, h in itertools.product(possible_lengths, possible_lengths)
                                if w >= h and w * h <= self.max_image_area and w / h <= self.max_ratio)

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
        return sum(len(b) // self.batch_size for b in self.bucket_data.values())

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
        max_bucket_len = max(len(b) for b in self.bucket_data.values())
        index_schedule = list(range(max_bucket_len))
        random.shuffle(index_schedule)

        bucket_len_table = {
            b: len(self.bucket_data[b]) for b in self.buckets
        }

        bucket_schedule = []
        for i, b in enumerate(self.buckets):
            bucket_schedule.extend(
                [i] * (bucket_len_table[b] // self.batch_size))

        random.shuffle(bucket_schedule)

        bucket_pos = {
            b: 0 for b in self.buckets
        }

        total_generated_by_bucket = {
            b: 0 for b in self.buckets
        }

        for bucket_index in bucket_schedule:
            b = self.buckets[bucket_index]
            i = bucket_pos[b]
            bucket_len = bucket_len_table[b]

            batch = []
            while len(batch) != self.batch_size:
                # advance in the schedule until we find an index that is contained in the bucket
                k = index_schedule[i]
                if k < bucket_len:
                    entry = self.bucket_data[b][k]
                    batch.append(entry)

                i += 1

            total_generated_by_bucket[b] += self.batch_size
            bucket_pos[b] = i
            yield [(idx, *b) for idx in batch]

    def fill_buckets(self):
        entries = self.store.entries_iterator()
        total_dropped = 0

        for entry, index in tqdm.tqdm(entries, total=len(self.store)):
            if not self._process_entry(entry, index):
                total_dropped += 1

        for b, values in self.bucket_data.items():
            # shuffle the entries for extra randomness and to make sure dropped elements are also random
            random.shuffle(values)

            # make sure the buckets have an exact number of elements for the batch
            to_drop = len(values) % self.batch_size
            self.bucket_data[b] = list(values[:len(values) - to_drop])
            total_dropped += to_drop

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

from torch.utils.data.sampler import Sampler,BatchSampler

class AspectBucketSampler(Sampler):
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
        return self.bucket.get_batch_count() // self.num_replicas


class AspectDataset(torch.utils.data.Dataset):
    def __init__(self, args, store: ImageStore, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, device: torch.device, ucg: float = 0.1):
        self.store = store
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device
        self.ucg = ucg
        self.args = args

        if type(self.text_encoder) is torch.nn.parallel.DistributedDataParallel:
            self.text_encoder = self.text_encoder.module

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.store)

    def __getitem__(self, item: Tuple[int, int, int]):
        return_dict = {'pixel_values': None, 'input_ids': None}

        image_file = self.store.get_image(item)

        return_dict['pixel_values'] = self.transforms(image_file)
        if random.random() > self.ucg:
            caption_file = self.store.get_caption(item)
        else:
            caption_file = ''

        return_dict['input_ids'] = caption_file
        return return_dict

    def collate_fn(self, examples):
        pixel_values = torch.stack([example['pixel_values']
                                   for example in examples if example is not None])
        pixel_values.to(memory_format=torch.contiguous_format).float()

        if self.args.extended_mode_chunks < 2:
            max_length = self.tokenizer.model_max_length - 2
            input_ids = [self.tokenizer([example['input_ids']], truncation=True, return_length=True, return_overflowing_tokens=False,
                                        padding=False, add_special_tokens=False, max_length=max_length).input_ids for example in examples if example is not None]
        else:
            max_length = self.tokenizer.model_max_length
            max_chunks = self.args.extended_mode_chunks
            input_ids = [self.tokenizer([example['input_ids']], truncation=True, return_length=True, return_overflowing_tokens=False, padding=False, add_special_tokens=False, max_length=(
                max_length * max_chunks) - (max_chunks * 2)).input_ids[0] for example in examples if example is not None]

        tokens = input_ids

        if self.args.extended_mode_chunks < 2:
            for i, x in enumerate(input_ids):
                for j, y in enumerate(x):
                    input_ids[i][j] = [self.tokenizer.bos_token_id, *y, *np.full(
                        (self.tokenizer.model_max_length - len(y) - 1), self.tokenizer.eos_token_id)]

            if self.args.clip_penultimate:
                input_ids = [self.text_encoder.text_model.final_layer_norm(self.text_encoder(torch.asarray(input_id).to(
                    self.device), output_hidden_states=True)['hidden_states'][-2])[0] for input_id in input_ids]
            else:
                input_ids = [self.text_encoder(torch.asarray(input_id).to(
                    self.device), output_hidden_states=True).last_hidden_state[0] for input_id in input_ids]
        else:
            max_standard_tokens = max_length - 2
            max_chunks = self.args.extended_mode_chunks
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
                                chunk.to(self.device), output_hidden_states=True)['hidden_states'][-2])
                        else:
                            z = self.text_encoder(
                                chunk.to(self.device), output_hidden_states=True).last_hidden_state
                    else:
                        if self.args.clip_penultimate:
                            z = torch.cat((z, self.text_encoder.text_model.final_layer_norm(self.text_encoder(
                                chunk.to(self.device), output_hidden_states=True)['hidden_states'][-2])), dim=-2)
                        else:
                            z = torch.cat((z, self.text_encoder(
                                chunk.to(self.device), output_hidden_states=True).last_hidden_state), dim=-2)
                input_ids = z
            else:
                for i, x in enumerate(input_ids):
                    input_ids[i] = [self.tokenizer.bos_token_id, *x, *np.full(
                        (self.tokenizer.model_max_length - len(x) - 1), self.tokenizer.eos_token_id)]
                if self.args.clip_penultimate:
                    input_ids = self.text_encoder.text_model.final_layer_norm(self.text_encoder(
                        torch.asarray(input_ids).to(self.device), output_hidden_states=True)['hidden_states'][-2])
                else:
                    input_ids = self.text_encoder(torch.asarray(input_ids).to(
                        self.device), output_hidden_states=True).last_hidden_state
        input_ids = torch.stack(tuple(input_ids))

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'tokens': tokens
        }


class ARBDataloader:
    def __init__(self, args, tokenizer, text_encoder, device, world_size, rank) -> None:
        self.store = ImageStore(args,args.train_data_dir)
        self.dataset = AspectDataset(
            args, self.store, tokenizer, text_encoder, device, ucg=args.ucg)
        self.bucket = AspectBucket(self.store, args.num_buckets, args.train_batch_size, args.bucket_side_min,
                              args.bucket_side_max, 64, args.resolution * args.resolution, 2.0)
        self.sampler =  AspectBucketSampler(
            bucket=self.bucket, num_replicas=world_size, rank=rank)
        print(f'STORE_LEN: {len(self.store)}')
        if args.output_bucket_info:
            print(self.bucket.get_bucket_info())
        self.train_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_sampler=self.sampler,
            #sampler = self.sampler,
            num_workers=0,
            collate_fn=self.dataset.collate_fn
        )

if __name__ == "__main__":
    pass