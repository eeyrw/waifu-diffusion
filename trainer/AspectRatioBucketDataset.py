import argparse
from bisect import bisect
import getpass
import math
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
from torch.utils.data.sampler import BatchSampler

from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageOps
from PIL.Image import Image as Img

from typing import Dict, List, Generator, Tuple
from scipy.interpolate import interp1d

from pillow_heif import register_heif_opener
register_heif_opener()

from Crypto.Cipher import AES
from Crypto.Util import Padding
import io

class CryptDatasetReader:
    def __init__(self):
        self.key=None
        self.keyBytes=None
        self.isCryptDs=False

    def accquireKey(self):
        if self.key is None:
            if 'DS_PASSWORD' not in os.environ.keys():
                # try:
                #     self.key = getpass.getpass()
                # except Exception as error:
                #     print('ERROR', error)
                # else:
                #     pass #print('Password entered:', key)
                pass
            else:
                self.key = os.environ['DS_PASSWORD']
                os.environ['DS_PASSWORD'] = self.key
                self.keyBytes = self.key.encode('utf8')
                self.keyBytes = Padding.pad(self.keyBytes, 16, style='pkcs7')


    def decrypt(self,f):
        self.accquireKey()
        tag = f.read(16)
        nonce = f.read(15)
        ciphertext = f.read()

        cipher = AES.new(self.keyBytes, AES.MODE_OCB, nonce=nonce)
        return cipher.decrypt_and_verify(ciphertext, tag)
    
    def readImage(self,fp):
        if not self.isCryptDs:
            return Image.open(fp)
        else:
            with open(fp, "rb") as f:
                img = Image.open(io.BytesIO(self.decrypt(f)))
            return img


    def readJson(self,fp):
        try:
            return json.load(fp)
        except Exception as e:
            self.isCryptDs = True
            fp.seek(0)
            return json.loads(self.decrypt(fp))
        

globalCryptDsReader = CryptDatasetReader()
    
def _sort_by_ratio(bucket: tuple) -> float:
    return bucket[0] / bucket[1]


def _sort_by_area(bucket: tuple) -> float:
    return bucket[0] * bucket[1]

def fitByVisualCenter(image, size, method=Image.Resampling.BICUBIC,visual_center=(0.5, 0.5)):
    """
    Returns a resized and cropped version of the image, cropped to the
    requested aspect ratio and size.

    This function was contributed by Kevin Cazabon.

    :param image: The image to resize and crop.
    :param size: The requested output size in pixels, given as a
                 (width, height) tuple.
    :param method: Resampling method to use. Default is
                   :py:attr:`~PIL.Image.Resampling.BICUBIC`.
                   See :ref:`concept-filters`.
    :param visual_center: current center (wc,hc) of image
    :return: An image.
    """

    # ensure centering is mutable
    visual_center = list(visual_center)

    live_size = (
        image.size[0],
        image.size[1],
    )

    # calculate the aspect ratio of the live_size
    live_size_ratio = live_size[0] / live_size[1]

    # calculate the aspect ratio of the output image
    output_ratio = size[0] / size[1]

    # figure out if the sides or top/bottom will be cropped off
    if live_size_ratio == output_ratio:
        # live_size is already the needed ratio
        crop_width = live_size[0]
        crop_height = live_size[1]
    elif live_size_ratio >= output_ratio:
        # live_size is wider than what's needed, crop the sides
        crop_width = output_ratio * live_size[1]
        crop_height = live_size[1]
    else:
        # live_size is taller than what's needed, crop the top and bottom
        crop_width = live_size[0]
        crop_height = live_size[0] / output_ratio

    # make the crop
    wc,hc = visual_center[0]*live_size[0],visual_center[1]*live_size[1]

    crop_left = wc - crop_width/2
    crop_right = wc + crop_width/2
    crop_top = hc - crop_height/2
    crop_bottom = hc + crop_height/2

    if crop_left<0:
        crop_left = 0
    if crop_right>live_size[0]:
        crop_left = live_size[0]-crop_width
    if crop_top<0:
        crop_top = 0
    if crop_bottom>live_size[1]:
        crop_top = live_size[1]-crop_height   


    crop = (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)

    # resize the image and return it
    return image.resize(size, method, box=crop)

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
            globalCryptDsReader.readImage(fp)
            return True
        except:
            print(f'WARNING: Image cannot be opened: {fp}')
            return False

    def __extended_validate(self, fp: str) -> bool:
        try:
            globalCryptDsReader.readImage(fp).load()
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

    def __no_migration(self, image_path: str, w: int, h: int, visual_center=(0.5,0.5)) -> Img:
        image = globalCryptDsReader.readImage(image_path)
        return fitByVisualCenter(
            image,
            (w, h),
            visual_center=visual_center,
            method=Image.Resampling.BICUBIC
        ).convert(mode='RGB')

    def __migration(self, image_path: str, w: int, h: int, visual_center=(0.5,0.5)) -> Img:
        filename = re.sub(r'\.[^/.]+$', '', os.path.split(image_path)[1])
        image = globalCryptDsReader.readImage(image_path)
        image = fitByVisualCenter(
            image,
            (w, h),
            visual_center=visual_center,
            method=Image.Resampling.BICUBIC
        ).convert(mode='RGB')

        if not globalCryptDsReader.isCryptDs:
            image.save(
                os.path.join(f'{self.__directory}', f'{filename}.jpg'),
                optimize=True
            )
        else:
            raise RuntimeError('Mirgation under crypt dataset is not allowed')

        return image

    def __no_op(self, image_path: str, w: int, h: int) -> Img:
        image = globalCryptDsReader.readImage(image_path)
        return image


class ImageStore:
    def __init__(self, args, data_dir: str) -> None:
        self.data_dir = data_dir
        if os.path.isdir(self.data_dir):
            imageInfoJsonPath = os.path.join(self.data_dir, 'ImageInfo.json')
        elif os.path.isfile(self.data_dir):
            imageInfoJsonPath = self.data_dir
            self.data_dir = os.path.dirname(imageInfoJsonPath)

        _, file_extension = os.path.splitext(imageInfoJsonPath)
        if file_extension == '.json':
            with open(imageInfoJsonPath, "rb") as f:
                self.imageInfoList = globalCryptDsReader.readJson(f)
        elif file_extension == '.jsonl':
            raise NotImplementedError('Not implement jsonl')     

        self.image_files = [os.path.join(
            self.data_dir, imageInfo['IMG']) for imageInfo in self.imageInfoList]
        self.validator = Validation(
            args.skip_validation,
            args.extended_validation
        ).validate

        self.resizer = Resize(args, args.resize, args.no_migration).resize

        self.image_files = [x for x in self.image_files if self.validator(x)]

        if not args.weighted_sample:
            self.imagesIdxList = list(range(len(self.imageInfoList)))
        else:
            self.imagesWeightList = np.asarray([imageInfo['WEIGHT'] for imageInfo in self.imageInfoList])
            self.imagesWeightList /= self.imagesWeightList.sum()
            self.resample_ds_by_weight()

    def __len__(self) -> int:
        return len(self.imagesIdxList)
    
    def resample_ds_by_weight(self):
        print('Resample DS')
        rawImagesIdxList = list(range(len(self.imageInfoList)))
        rng = np.random.default_rng()
        resampleList = rng.choice(rawImagesIdxList,len(self.imageInfoList)*20,replace=True,p=self.imagesWeightList)
        self.imagesIdxList = resampleList.tolist()

    # iterator returns images as PIL images and their index in the store
    def entries_iterator(self) -> Generator[Tuple[Dict, int], None, None]:
        for f in range(len(self)):
            remapIdx = self.imagesIdxList[f]
            yield self.imageInfoList[remapIdx], f

    # get image by index
    def get_image(self, ref: Tuple[int, int, int]) -> Img:
        remapIdx = self.imagesIdxList[ref[0]]
        if 'V_CENTER' in self.imageInfoList[remapIdx].keys():
            visual_center = self.imageInfoList[remapIdx]['V_CENTER']
        else:
            visual_center = (0.5,0.5)
        return self.resizer(
            self.image_files[remapIdx],
            ref[1],
            ref[2],
            visual_center=visual_center
        )

    # gets caption by removing the extension from the filename and replacing it with .txt
    def get_caption(self, ref: Tuple[int, int, int]) -> str:
        qualityDescList = []
        extraTags = []
        isNegativeSample = False
        remapIdx = self.imagesIdxList[ref[0]]

        # if 'LBL' in self.imageInfoList[remapIdx].keys() and self.imageInfoList[remapIdx]['LBL']!=-1:
        #     extraTags.append(f'pose_{self.imageInfoList[remapIdx]['LBL']}')

        # if 'A_EAT' in self.imageInfoList[remapIdx].keys():
        #     A = self.imageInfoList[remapIdx]['A_EAT']
        #     if A>5.5:
        #         qualityDescList.append('masterpiece')
        #     elif A<3:
        #         qualityDescList.append('bad art')
        #         # isNegativeSample = True

        # if 'Q512' in self.imageInfoList[remapIdx].keys():
        #     Q = self.imageInfoList[remapIdx]['Q512']
        #     if Q>65:
        #         qualityDescList.append('high res,best quality')
        #     elif Q<40:
        #         qualityDescList.append('low res,low quality')
        #         isNegativeSample = True

        caption_key = random.choice(['HQ_CAP','DBRU_TAG'])
        if caption_key not in self.imageInfoList[remapIdx].keys() and caption_key=='HQ_CAP':
            caption_key = 'DBRU_TAG'
        if caption_key not in self.imageInfoList[remapIdx].keys() and caption_key=='DBRU_TAG':
            caption_key = 'HQ_CAP'

        if caption_key in self.imageInfoList[remapIdx].keys():
            captions = self.imageInfoList[remapIdx][caption_key]
        else:
            captions = None
            #print(self.imageInfoList[remapIdx]['IMG'])
        if captions is None:
            caption = ""
        else:
            if isinstance(captions, list):
                caption = random.choice(captions)
            elif isinstance(captions, str):
                caption = captions

            if caption_key == 'DBRU_TAG':
                tagList = caption.split(',')
                tagList = tagList+extraTags
                random.shuffle(tagList)
                caption = ','.join(tagList)
            # if 'artist' in self.imageInfoList[ref[0]].keys():
            #     caption = 'by artist '+ self.imageInfoList[ref[0]]['artist'] + caption
            # if 'style' in self.imageInfoList[ref[0]].keys():
            #     caption = 'in style of '+ self.imageInfoList[ref[0]]['style'] + caption
            
        # if random.random() > 0.7:    
        #     caption = caption+','+','.join(qualityDescList)
            
        #random.shuffle(qualityDescList)    
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
                 bucket_mode: str = 'multiscale',
                 max_image_area: int = 512 * 768,
                 multi_resolution=[512,640,768],
                 max_ratio: float = 2):

        self.requested_bucket_count = num_buckets
        self.bucket_length_min = bucket_side_min
        self.bucket_length_max = bucket_side_max
        self.bucket_increment = bucket_side_increment
        self.bucket_mode = bucket_mode
        self.max_image_area = max_image_area
        self.multi_resolution = multi_resolution
        self.batch_size = batch_size
        self.total_dropped = 0

        if max_ratio <= 0:
            self.max_ratio = float('inf')
        else:
            self.max_ratio = max_ratio

        self.store = store
        self.buckets = {}
        self._bucket_ratios = {}
        self.bucket_data: Dict[int,Dict[tuple, List[int]]] = dict()
        self.init_buckets()
        self._build_bucket_lookup()
        self.fill_buckets()

    def get_buckets(self,mode,maxPixelNum):
        if mode == 'maxfit':
            # https://blog.novelai.net/novelai-improvements-on-stable-diffusion-e10d38db82ac
            # ● Set the width to 256.
            # ● While the width is less than or equal to 1024:
            # • Find the largest height such that height is less than or equal to 1024 and that width multiplied by height is less than or equal to 512 * 768.
            # • Add the resolution given by height and width as a bucket.
            # • Increase the width by 64.
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
        for resolution in self.multi_resolution:
            possible_buckets = self.get_buckets(self.bucket_mode,resolution*resolution)

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

            self.buckets[resolution] = buckets

            # cache the bucket ratios and the interpolator that will be used for calculating the best bucket later
            # the interpolator makes a 1d piecewise interpolation where the input (x-axis) is the bucket ratio,
            # and the output is the bucket index in the self.buckets array
            # to find the best fit we can just round that number to get the index
            self._bucket_ratios[resolution] = [w / h for w, h in buckets]

            for b in buckets:
                self.bucket_data.setdefault(resolution,{b:[]}).update({b:[]})


    def get_batch_count(self):
        return sum(sum(len(b) // self.batch_size for b in bucket_data_single.values()) \
                   for bucket_data_single in self.bucket_data.values())

    def get_bucket_info(self):
        return json.dumps({"buckets": self.buckets, "bucket_ratios": self._bucket_ratios})

    def get_batch_iterator(self, rng: "random.Random" = None, rank: int = 0, num_replicas: int = 1) -> Generator[Tuple[Tuple[int, int, int]], None, None]:
        """
        Lazy generator that yields batches [(idx,w,h), ...] distributed by rank.
        """
        if rng is None:
            rng = random.Random()

        # merge bucket_data across resolutions
        bucket_data_merged = {}
        buckets_merged = set()
        for res, buckets_single in self.buckets.items():
            buckets_merged.update(buckets_single)
        buckets_merged = list(buckets_merged)

        for res, bucket_data_single in self.bucket_data.items():
            for b, idcs in bucket_data_single.items():
                bucket_data_merged.setdefault(b, []).extend(idcs)

        # lengths
        bucket_len_table = {b: len(bucket_data_merged.get(b, [])) for b in buckets_merged}

        # build index schedule and bucket schedule
        max_bucket_len = max(bucket_len_table.values(), default=0)
        index_schedule = list(range(max_bucket_len))
        rng.shuffle(index_schedule)

        bucket_schedule = []
        for i, b in enumerate(buckets_merged):
            bucket_schedule.extend([i] * (bucket_len_table[b] // self.batch_size))
        rng.shuffle(bucket_schedule)

        bucket_pos = {b: 0 for b in buckets_merged}

        # iterate bucket_schedule lazily
        for i, bucket_index in enumerate(bucket_schedule):
            # 只保留属于当前 rank 的 batch
            if (i % num_replicas) != rank:
                continue
            b = buckets_merged[bucket_index]
            i_pos = bucket_pos[b]
            batch = []
            while len(batch) != self.batch_size:
                if i_pos >= len(index_schedule):
                    break
                k = index_schedule[i_pos]
                if k < bucket_len_table[b]:
                    entry = bucket_data_merged[b][k]
                    batch.append(entry)
                i_pos += 1
            bucket_pos[b] = i_pos
            if len(batch) == self.batch_size:
                yield [(idx, *b) for idx in batch]


    def fill_buckets(self):
        entries = self.store.entries_iterator()
        total_dropped = 0

        for entry, index in tqdm.tqdm(entries, total=len(self.store)):
            if not self._process_entry(entry, index):
                total_dropped += 1

        for res,bucket_data_single in self.bucket_data.items():
            for b, values in bucket_data_single.items():
                # 不在 init 时 shuffle，延迟到迭代时使用可控 rng 进行 shuffle
                to_drop = len(values) % self.batch_size
                # 保证被丢弃的元素是末尾元素（可考虑用 rng 在迭代时随机丢弃）
                self.bucket_data[res][b] = list(values[:len(values) - to_drop])
                total_dropped += to_drop

        self.total_dropped = total_dropped


    # 在 AspectBucket._build_bucket_lookup 中添加
    def _build_bucket_lookup(self):
        """
        构建平铺 bucket 列表，并缓存 ratio 以加速 _process_entry
        """
        self._all_buckets = [(res, bw, bh) for res, bucket_list in self.buckets.items() for bw, bh in bucket_list]
        # 预计算 ratio
        self._bucket_ratios_flat = [(res, bw, bh, bw / bh) for res, bw, bh in self._all_buckets]

        self._bucket_to_res = {b:res for res, bucket_list in self.buckets.items() for b in bucket_list}

    def _process_entry(self, entry: Dict, index: int, max_downscale: float = 2.0, rng: random.Random = None) -> bool:
        """
        Process a single image entry and assign it to a bucket.
        Supports downsampling only: bucket dimensions <= original image.
        rng: 用于保证 epoch 可重复的随机数
        """
        if rng is None:
            rng = random.Random()
        
        orig_w, orig_h = entry['W'], entry['H']
        aspect = orig_w / orig_h

        # 丢弃过极端的长宽比
        if aspect > self.max_ratio or (1 / aspect) > self.max_ratio:
            return False

        # 找出所有满足条件的 candidate buckets
        candidate_buckets = [
            (res, bw, bh, abs(aspect - r))
            for res, bw, bh, r in self._bucket_ratios_flat
            if bw <= orig_w and bh <= orig_h  # 分辨率必须 <= 原图
            and orig_w/bw <= max_downscale and orig_h/bh <= max_downscale
        ]

        if not candidate_buckets:
            return False

        # 先按长宽比差排序，选 top-k
        candidate_buckets.sort(key=lambda x: x[3])
        top_k = min(3, len(candidate_buckets))

        for best_res, bw, bh, _ in candidate_buckets[:top_k]:
            self.bucket_data[best_res][(bw, bh)].append(index)
        return True


class AspectBucketSampler(BatchSampler):
    def __init__(self, 
                 bucket: AspectBucket,
                 num_replicas: int = 1, 
                 rank: int = 0,
                 batch_size: int = 1,
                 drop_last: bool = False,
                 base_seed: int = 42):
        # Note: we don't call super().__init__ because we implement __iter__ ourselves
        self.bucket = bucket
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.base_seed = int(base_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        """Call this at the start of each epoch (DDP convention)"""
        self.epoch = int(epoch)

    def _get_rng_for_epoch(self):
        # deterministic per (base_seed, epoch)
        return random.Random(self.base_seed + self.epoch)

    def __iter__(self):
        rng = self._get_rng_for_epoch()
        # 直接把 rank 和 num_replicas 传入 bucket 的迭代器，内部完成分配
        yield from self.bucket.get_batch_iterator(rng=rng, rank=self.rank, num_replicas=self.num_replicas)

    def __len__(self):
        # compute how many batches this sampler will yield for this rank
        total = self.bucket.get_batch_count()
        per = total // self.num_replicas
        remainder = total % self.num_replicas
        # distribute the remainder to first `remainder` ranks
        if self.rank < remainder:
            per += 1
        return per


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
            #torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.store)

    def __getitem__(self, item: Tuple[int, int, int]):
        return_dict = {'pixel_values': None, 'input_text': None}

        image_file = self.store.get_image(item)

        return_dict['pixel_values'] = self.transforms(image_file)
        if random.random() > self.ucg:
            caption_file = self.store.get_caption(item)
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
            #'tokens': tokens
        }


class ARBDataloader:
    def __init__(self, args, tokenizer, text_encoder, device, world_size, rank) -> None:
        self.store = ImageStore(args,args.train_data_dir)

        self.bucket = AspectBucket(self.store, args.num_buckets, args.train_batch_size, args.bucket_side_min,
                              args.bucket_side_max, 64, args.bucket_mode,args.resolution * args.resolution, args.multi_resolution,2.0)
        self.sampler =  AspectBucketSampler(
            bucket=self.bucket, num_replicas=world_size, rank=rank)
        self.dataset = AspectDataset(
            args, self.store, tokenizer, text_encoder, device, ucg=args.ucg)
        print(f'STORE_LEN: {len(self.store)}')
        if args.output_bucket_info:
            print(self.bucket.get_bucket_info())
        self.train_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_sampler=self.sampler,
            num_workers=args.dataloader_num_workers,
            collate_fn=self.dataset.collate_fn
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default='xxx',
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    parser.add_argument(
        "--multi_resolution",
        type=lambda x:list(map(int, x.split(','))),
        default=[512,574,640,704,768,832,896,960,1024],
        help=(
            'The multiple resolution bucket'
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    def bool_t(x): return x.lower() in ['true', 'yes', '1']
    parser.add_argument('--num_buckets', type=int, default=16,
                        help='The number of buckets.')
    parser.add_argument('--bucket_mode', type=str, default='maxfit',
                        help='multiscale|maxfit')
    parser.add_argument('--bucket_side_min', type=int, default=256,
                        help='The minimum side length of a bucket.')
    parser.add_argument('--bucket_side_max', type=int, default=1280,
                        help='The maximum side length of a bucket.')
    parser.add_argument('--ucg', type=float, default=0.1,
                        help='Percentage chance of dropping out the text condition per batch. Ranges from 0.0 to 1.0 where 1.0 means 100% text condition dropout.')  # 10% dropout probability
    parser.add_argument('--shuffle', dest='shuffle', type=bool_t,
                        default='True', help='Shuffle dataset')
    parser.add_argument('--output_bucket_info', type=bool_t,
                        default='False', help='Outputs bucket information and exits')
    parser.add_argument('--resize', type=bool_t, default='True',
                        help="Resizes dataset's images to the appropriate bucket dimensions.")
    parser.add_argument('--extended_validation', type=bool_t, default='False',
                        help='Perform extended validation of images to catch truncated or corrupt images.')
    parser.add_argument('--no_migration', type=bool_t, default='True',
                        help='Do not perform migration of dataset while the `--resize` flag is active. Migration creates an adjacent folder to the dataset with <dataset_dirname>_cropped.')
    parser.add_argument('--skip_validation', type=bool_t, default='True',
                        help='Skip validation of images, useful for speeding up loading of very large datasets that have already been validated.')

    parser.add_argument('--clip_penultimate', type=bool_t, default='False',
                        help='Use penultimate CLIP layer for text embedding')
    parser.add_argument('--extended_mode_chunks', type=int, default=0,
                        help='Enables extended mode for tokenization with given amount of maximum chunks. Values < 2 disable.')
    parser.add_argument('--local_files_only', type=bool_t, default='False',
                        help='Do not connect to HF')
    parser.add_argument('--weighted_sample', type=bool_t, default='False',
                        help='Use weighted sample')
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    args = parser.parse_args()

    arbDataloader = ARBDataloader(args,None,None,'cpu',1,0)
    from torchvision import utils
    for i,p in tqdm.tqdm(enumerate(arbDataloader.train_dataloader)):
        pixel_value = p['pixel_values']
        with open(f'arbTestoutput/{i}.txt','w') as f:
            for input_text in p['input_texts']:
                f.write(input_text+'\n')
        pixel_value = pixel_value/2+0.5
        utils.save_image(pixel_value,f'arbTestoutput/{i}.webp')

        #print(p)