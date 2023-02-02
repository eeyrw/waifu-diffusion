# Install bitsandbytes:
# `nvcc --version` to get CUDA version.
# `pip install -i https://test.pypi.org/simple/ bitsandbytes-cudaXXX` to install for current CUDA.
# Example Usage:
# Single GPU: torchrun --nproc_per_node=1 trainer/diffusers_trainer.py --model="CompVis/stable-diffusion-v1-4" --run_name="liminal" --dataset="liminal-dataset" --hf_token="hf_blablabla" --bucket_side_min=64 --use_8bit_adam=True --gradient_checkpointing=True --batch_size=1 --fp16=True --image_log_steps=250 --epochs=20 --resolution=768 --use_ema=True
# Multiple GPUs: torchrun --nproc_per_node=N trainer/diffusers_trainer.py --model="CompVis/stable-diffusion-v1-4" --run_name="liminal" --dataset="liminal-dataset" --hf_token="hf_blablabla" --bucket_side_min=64 --use_8bit_adam=True --gradient_checkpointing=True --batch_size=10 --fp16=True --image_log_steps=250 --epochs=20 --resolution=768 --use_ema=True

import argparse
import socket
import torch
import torchvision
import transformers
import diffusers
import os
import glob
import random
import tqdm
import resource
import psutil
import pynvml
import wandb
import gc
import time
import itertools
import numpy as np
import json
import re
import traceback
import shutil
from safetensors.torch import load_file

try:
    pynvml.nvmlInit()
except pynvml.nvml.NVMLError_LibraryNotFound:
    pynvml = None

from typing import Iterable, Optional
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.optimization import get_scheduler
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageOps
from PIL.Image import Image as Img

from typing import Dict, List, Generator, Tuple
from scipy.interpolate import interp1d

torch.backends.cuda.matmul.allow_tf32 = True

# defaults should be good for everyone
# TODO: add custom VAE support. should be simple with diffusers
bool_t = lambda x: x.lower() in ['true', 'yes', '1']
parser = argparse.ArgumentParser(description='Stable Diffusion Finetuner')
parser.add_argument('--model', type=str, default=None, required=True, help='The name of the model to use for finetuning. Could be HuggingFace ID or a directory')
parser.add_argument('--resume', type=str, default=None, help='The path to the checkpoint to resume from. If not specified, will create a new run.')
parser.add_argument('--run_name', type=str, default=None, required=True, help='Name of the finetune run.')
parser.add_argument('--dataset', type=str, default=None, required=True, help='The path to the dataset to use for finetuning.')
parser.add_argument('--num_buckets', type=int, default=16, help='The number of buckets.')
parser.add_argument('--bucket_side_min', type=int, default=256, help='The minimum side length of a bucket.')
parser.add_argument('--bucket_side_max', type=int, default=768, help='The maximum side length of a bucket.')
parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--use_ema', type=bool_t, default='False', help='Use EMA for finetuning')
parser.add_argument('--ucg', type=float, default=0.1, help='Percentage chance of dropping out the text condition per batch. Ranges from 0.0 to 1.0 where 1.0 means 100% text condition dropout.') # 10% dropout probability
parser.add_argument('--gradient_checkpointing', dest='gradient_checkpointing', type=bool_t, default='False', help='Enable gradient checkpointing')
parser.add_argument('--use_8bit_adam', dest='use_8bit_adam', type=bool_t, default='False', help='Use 8-bit Adam optimizer')
parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam beta1')
parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam beta2')
parser.add_argument('--adam_weight_decay', type=float, default=1e-2, help='Adam weight decay')
parser.add_argument('--adam_epsilon', type=float, default=1e-08, help='Adam epsilon')
parser.add_argument('--lr_scheduler', type=str, default='constant', help='Learning rate scheduler [`cosine`, `linear`, `constant`]')
parser.add_argument('--lr_scheduler_warmup', type=float, default=0.05, help='Learning rate scheduler warmup steps. This is a percentage of the total number of steps in the training run. 0.1 means 10 percent of the total number of steps.')
parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator, this is to be used for reproduceability purposes.')
parser.add_argument('--output_path', type=str, default='./output', help='Root path for all outputs.')
parser.add_argument('--save_steps', type=int, default=500, help='Number of steps to save checkpoints at.')
parser.add_argument('--resolution', type=int, default=512, help='Image resolution to train against. Lower res images will be scaled up to this resolution and higher res images will be scaled down.')
parser.add_argument('--shuffle', dest='shuffle', type=bool_t, default='True', help='Shuffle dataset')
parser.add_argument('--hf_token', type=str, default=None, required=False, help='A HuggingFace token is needed to download private models for training.')
parser.add_argument('--project_id', type=str, default='diffusers', help='Project ID for reporting to WandB')
parser.add_argument('--fp16', dest='fp16', type=bool_t, default='False', help='Train in mixed precision')
parser.add_argument('--image_log_steps', type=int, default=500, help='Number of steps to log images at.')
parser.add_argument('--image_log_amount', type=int, default=4, help='Number of images to log every image_log_steps')
parser.add_argument('--image_log_inference_steps', type=int, default=50, help='Number of inference steps to use to log images.')
parser.add_argument('--image_log_scheduler', type=str, default="PNDMScheduler", help='Number of inference steps to use to log images.')
parser.add_argument('--clip_penultimate', type=bool_t, default='False', help='Use penultimate CLIP layer for text embedding')
parser.add_argument('--output_bucket_info', type=bool_t, default='False', help='Outputs bucket information and exits')
parser.add_argument('--resize', type=bool_t, default='False', help="Resizes dataset's images to the appropriate bucket dimensions.")
parser.add_argument('--use_xformers', type=bool_t, default='False', help='Use memory efficient attention')
parser.add_argument('--wandb', dest='enablewandb', type=bool_t, default='True', help='Enable WeightsAndBiases Reporting')
parser.add_argument('--inference', dest='enableinference', type=bool_t, default='True', help='Enable Inference during training (Consumes 2GB of VRAM)')
parser.add_argument('--extended_validation', type=bool_t, default='False', help='Perform extended validation of images to catch truncated or corrupt images.')
parser.add_argument('--no_migration', type=bool_t, default='False', help='Do not perform migration of dataset while the `--resize` flag is active. Migration creates an adjacent folder to the dataset with <dataset_dirname>_cropped.')
parser.add_argument('--skip_validation', type=bool_t, default='False', help='Skip validation of images, useful for speeding up loading of very large datasets that have already been validated.')
parser.add_argument('--extended_mode_chunks', type=int, default=0, help='Enables extended mode for tokenization with given amount of maximum chunks. Values < 2 disable.')
parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
parser.add_argument('--model_cache_dir', type=str, default=None, required=True,
                    help='The name of the model cache directory')
parser.add_argument('--debug', dest='debug', type=bool_t, default='False', help='Enable WeightsAndBiases Reporting')
parser.add_argument('--train_from_scratch', dest='train_from_scratch', type=bool_t, default='False', help='Whether train from scratch')
parser.add_argument('--use_grad_clip', dest='use_grad_clip', type=bool_t, default='False', help='Whether train from scratch')


args = parser.parse_args()

def setup():
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

def get_gpu_ram() -> str:
    """
    Returns memory usage statistics for the CPU, GPU, and Torch.

    :return:
    """
    gpu_str = ""
    torch_str = ""
    try:
        cudadev = torch.cuda.current_device()
        nvml_device = pynvml.nvmlDeviceGetHandleByIndex(cudadev)
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_device)
        gpu_total = int(gpu_info.total / 1E6)
        gpu_free = int(gpu_info.free / 1E6)
        gpu_used = int(gpu_info.used / 1E6)
        gpu_str = f"GPU: (U: {gpu_used:,}mb F: {gpu_free:,}mb " \
                  f"T: {gpu_total:,}mb) "
        torch_reserved_gpu = int(torch.cuda.memory.memory_reserved() / 1E6)
        torch_reserved_max = int(torch.cuda.memory.max_memory_reserved() / 1E6)
        torch_used_gpu = int(torch.cuda.memory_allocated() / 1E6)
        torch_max_used_gpu = int(torch.cuda.max_memory_allocated() / 1E6)
        torch_str = f"TORCH: (R: {torch_reserved_gpu:,}mb/"  \
                    f"{torch_reserved_max:,}mb, " \
                    f"A: {torch_used_gpu:,}mb/{torch_max_used_gpu:,}mb)"
    except AssertionError:
        pass
    cpu_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1E3 +
                     resource.getrusage(
                         resource.RUSAGE_CHILDREN).ru_maxrss / 1E3)
    cpu_vmem = psutil.virtual_memory()
    cpu_free = int(cpu_vmem.free / 1E6)
    return f"CPU: (maxrss: {cpu_maxrss:,}mb F: {cpu_free:,}mb) " \
           f"{gpu_str}" \
           f"{torch_str}"


class LatentStore:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        latentInfoJsonPath = os.path.join(self.data_dir, 'LatentInfo.json')
        with open(latentInfoJsonPath, "r") as f:
            self.latentInfoDict = json.load(f)
        self.unconditionalTxtEmb = load_file(os.path.join(self.data_dir,'UnconditionalTxtEmb.safetensors'))['unconditionalTxtEmb']

    def __len__(self) -> int:
        return self.latentInfoDict['NumSamples']

    # get image by index
    def get_latent(self, ref: Tuple[int, int, int]) -> Dict[str, torch.Tensor]:
        fileName = os.path.join(self.data_dir,'%dx%d/%d.safetensors'%(ref[1],ref[2],ref[0]))
        try:
            return load_file(fileName)
        except FileNotFoundError:
            print(fileName)
            raise RuntimeError('Fail to load file.')
    
    def getUnconditionalTxtEmb(self):
        return self.unconditionalTxtEmb

    def get_bucket_data(self) -> Dict[tuple, List[int]]:
        bucket_data = {}
        for bucketName,bucketIndexList in self.latentInfoDict['LatentDict'].items():
            w,h = bucketName.split('x',1)
            w,h = int(w),int(h)
            bucket_data.setdefault((w,h),[]).extend(bucketIndexList)
        return bucket_data



# ====================================== #
# Bucketing code stolen from hasuwoof:   #
# https://github.com/hasuwoof/huskystack #
# ====================================== #

class LatentBucket:
    def __init__(self, store: LatentStore,
                 batch_size: int):

        self.batch_size = batch_size
        self.total_dropped = 0


        self.store = store
        self.bucket_data: Dict[tuple, List[int]] = dict()
        self.fill_buckets()

    def get_batch_count(self):
        return sum(len(b) // self.batch_size for b in self.bucket_data.values())

    def get_bucket_info(self):
        return json.dumps({ "buckets": self.buckets})

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
            bucket_schedule.extend([i] * (bucket_len_table[b] // self.batch_size))

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
        self.bucket_data = self.store.get_bucket_data()
        self.buckets = []
        for w,h in self.bucket_data:
            self.buckets.append((h,w))


class LatentBucketSampler(torch.utils.data.Sampler):
    def __init__(self, bucket: LatentBucket, num_replicas: int = 1, rank: int = 0):
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

class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, store: LatentStore, device: torch.device, ucg: float = 0.1):
        self.store = store
        self.device = device
        self.ucg = ucg

    def __len__(self):
        return len(self.store)

    def __getitem__(self, item: Tuple[int, int, int]):
        return_dict = {}
        latentDict = self.store.get_latent(item)
        if random.random() > self.ucg:
            return_dict['txtEmb'] = latentDict['txtEmb']
        else:
            return_dict['txtEmb'] = self.store.getUnconditionalTxtEmb()
        return_dict['imgLatent'] = latentDict['imgLatent']
        
        return return_dict

    def collate_fn(self, examples):
        imgLatents = torch.stack([example['imgLatent']
                                   for example in examples if example is not None])
        imgLatents.to(memory_format=torch.contiguous_format).float()
        txtEmbs = torch.stack([example['txtEmb']
                                   for example in examples if example is not None])
        txtEmbs.to(memory_format=torch.contiguous_format).float()
        return {
            'imgLatents': imgLatents,
            'txtEmbs': txtEmbs,
        }
class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        device = self.parameters.device
        sample_device = "cpu" if device.type == "mps" else device
        sample = torch.randn(self.mean.shape, generator=generator, device=sample_device)
        # make sure sample is on the same device as the parameters and has same dtype
        sample = sample.to(device=device, dtype=self.parameters.dtype)
        x = self.mean + self.std * sample
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean
# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    # From CompVis LitEMA implementation
    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

        del self.collected_params
        gc.collect()

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]

def main():
    rank = get_rank()
    world_size = get_world_size()
    torch.cuda.set_device(rank)

    if rank == 0:
        os.makedirs(args.output_path, exist_ok=True)
        
        mode = 'disabled'
        if args.enablewandb:
            mode = 'online'
        if args.hf_token is not None:
            os.environ['HF_API_TOKEN'] = args.hf_token
            args.hf_token = None
        run = wandb.init(project=args.project_id, name=args.run_name, config=vars(args), dir=args.output_path+'/wandb', mode=mode)

        # Inform the user of host, and various versions -- useful for debugging issues.
        print("RUN_NAME:", args.run_name)
        print("HOST:", socket.gethostname())
        print("CUDA:", torch.version.cuda)
        print("TORCH:", torch.__version__)
        print("TRANSFORMERS:", transformers.__version__)
        print("DIFFUSERS:", diffusers.__version__)
        print("MODEL:", args.model)
        print("FP16:", args.fp16)
        print("RESOLUTION:", args.resolution)


    if args.hf_token is not None:
        print('It is recommended to set the HF_API_TOKEN environment variable instead of passing it as a command line argument since WandB will automatically log it.')
    else:
        try:
            args.hf_token = os.environ['HF_API_TOKEN']
            print("HF Token set via enviroment variable")
        except Exception:
            print("No HF Token detected in arguments or enviroment variable, setting it to none (as in string)")
            args.hf_token = "none"

    device = torch.device('cuda')

    print("DEVICE:", device)

    # setup fp16 stuff
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print('RANDOM SEED:', args.seed)

    if args.resume:
        args.model = args.resume

    if args.model[0] == '.':
        args.model = os.path.join(
            args.model_cache_dir, args.model[1:])

    if args.train_from_scratch:
        unet = UNet2DConditionModel.from_config({
            "_class_name": "UNet2DConditionModel",
            "_diffusers_version": "0.11.1",
            "_name_or_path": "stabilityai/stable-diffusion-2-1",
            "act_fn": "silu",
            "attention_head_dim": [
                5,
                10,
                20,
                20
            ],
            "block_out_channels": [
                320,
                640,
                1280,
                1280
            ],
            "center_input_sample": False,
            "class_embed_type": None,
            "cross_attention_dim": 1024,
            "down_block_types": [
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D"
            ],
            "downsample_padding": 1,
            "dual_cross_attention": False,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "in_channels": 4,
            "layers_per_block": 2,
            "mid_block_scale_factor": 1,
            "mid_block_type": "UNetMidBlock2DCrossAttn",
            "norm_eps": 1e-05,
            "norm_num_groups": 32,
            "num_class_embeds": None,
            "only_cross_attention": False,
            "out_channels": 4,
            "resnet_time_scale_shift": "default",
            "sample_size": 96,
            "up_block_types": [
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D"
            ],
            "upcast_attention": True,
            "use_linear_projection": True
            }
        )
    else:
        tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder='tokenizer', use_auth_token=args.hf_token,cache_dir=args.model_cache_dir)
        text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder='text_encoder', use_auth_token=args.hf_token,cache_dir=args.model_cache_dir)
        vae = AutoencoderKL.from_pretrained(args.model, subfolder='vae', use_auth_token=args.hf_token,cache_dir=args.model_cache_dir)
        unet = UNet2DConditionModel.from_pretrained(args.model, subfolder='unet', use_auth_token=args.hf_token,cache_dir=args.model_cache_dir)

    # Freeze vae and text_encoder
    # vae.requires_grad_(False)
    # if not args.train_text_encoder:
    #     text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        # if args.train_text_encoder:
        #     text_encoder.gradient_checkpointing_enable()

    if args.use_xformers:
        unet.set_use_memory_efficient_attention_xformers(True)

    # "The “safer” approach would be to move the model to the device first and create the optimizer afterwards."
    weight_dtype = torch.float16 if args.fp16 else torch.float32

    # move models to device
    # vae = vae.to(device, dtype=weight_dtype) 
    unet = unet.to(device, dtype=torch.float32)
    # text_encoder = text_encoder.to(device, dtype=weight_dtype if not args.train_text_encoder else torch.float32)

    unet = torch.nn.parallel.DistributedDataParallel(
        unet,
        device_ids=[rank],
        output_device=rank,
        gradient_as_bucket_view=True
    )

    # if args.train_text_encoder:
    #     text_encoder = torch.nn.parallel.DistributedDataParallel(
    #         text_encoder,
    #         device_ids=[rank],
    #         output_device=rank,
    #         gradient_as_bucket_view=True
    #     )

    if args.use_8bit_adam: # Bits and bytes is only supported on certain CUDA setups, so default to regular adam if it fails.
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except:
            print('bitsandbytes not supported, using regular Adam optimizer')
            optimizer_cls = torch.optim.AdamW
    else:
        optimizer_cls = torch.optim.AdamW

    """
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )
    """

    optimizer_parameters = unet.parameters() #if not args.train_text_encoder else itertools.chain(unet.parameters(), text_encoder.parameters())

    # Create distributed optimizer
    from torch.distributed.optim import ZeroRedundancyOptimizer
    optimizer = ZeroRedundancyOptimizer(
        optimizer_parameters,
        optimizer_class=optimizer_cls,
        parameters_as_bucket_view=True,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )


    noise_scheduler = DDPMScheduler.from_pretrained(
        args.model,
        subfolder='scheduler',
        use_auth_token=args.hf_token,
    )

    # load dataset
    store = LatentStore(args.dataset)
    dataset = LatentDataset(store, device, ucg=args.ucg)
    bucket = LatentBucket(store, args.batch_size)
    sampler = LatentBucketSampler(bucket=bucket, num_replicas=world_size, rank=rank)

    print(f'STORE_LEN: {len(store)}')

    if args.output_bucket_info:
        print(bucket.get_bucket_info())
        exit(0)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )
    
    # Migrate dataset
    if args.resize and not args.no_migration:
        for _, batch in enumerate(train_dataloader):
            continue
        print(f"Completed resize and migration to '{args.dataset}_cropped' please relaunch the trainer without the --resize argument and train on the migrated dataset.")
        exit(0)

    # create ema
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters())

    print(get_gpu_ram())

    num_steps_per_epoch = len(train_dataloader)
    progress_bar = tqdm.tqdm(range(args.epochs * num_steps_per_epoch), desc="Total Steps", leave=False)
    global_step = 0

    if args.resume:
        target_global_step = int(args.resume.split('_')[-1])
        print(f'resuming from {args.resume}...')

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(args.lr_scheduler_warmup * num_steps_per_epoch * args.epochs),
        num_training_steps=args.epochs * num_steps_per_epoch,
        #last_epoch=(global_step // num_steps_per_epoch) - 1,
    )

    def save_checkpoint(global_step):
        if rank == 0:
            if args.use_ema:
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())
            pipeline = StableDiffusionPipeline(
                text_encoder=text_encoder if type(text_encoder) is not torch.nn.parallel.DistributedDataParallel else text_encoder.module,
                vae=vae,
                unet=unet.module,
                tokenizer=tokenizer,
                scheduler=PNDMScheduler.from_pretrained(args.model, subfolder="scheduler", use_auth_token=args.hf_token),
                safety_checker=None,#StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
                feature_extractor=None,#CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
            )
            print(f'saving checkpoint to: {args.output_path}/{args.run_name}_{global_step}')
            pipeline.save_pretrained(f'{args.output_path}/{args.run_name}_{global_step}')

            if args.use_ema:
                ema_unet.restore(unet.parameters())

    # train!
    try:
        loss = torch.tensor(0.0, device=device, dtype=weight_dtype)
        for epoch in range(args.epochs):
            unet.train()
            # if args.train_text_encoder:
            #     text_encoder.train()
            for _, batch in enumerate(train_dataloader):
                if args.resume and global_step < target_global_step:
                    if rank == 0:
                        progress_bar.update(1)
                    global_step += 1
                    continue
                
                b_start = time.perf_counter()
                #latents = vae.encode(batch['pixel_values'].to(device, dtype=weight_dtype)).latent_dist.sample()
                #latents = latents * 0.18215
                latents = batch['imgLatents'].to(device, dtype=weight_dtype)
                latents = DiagonalGaussianDistribution(latents).sample()
                latents = latents * 0.18215

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the embedding for conditioning
                encoder_hidden_states = batch['txtEmbs'].to(device, dtype=weight_dtype)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

                if not args.train_text_encoder:
                    with unet.join():
                        # Predict the noise residual and compute loss
                        with torch.autocast('cuda', enabled=args.fp16):
                            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                            
                        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                        # backprop and update
                        # scaler.scale(loss).backward()
                        # torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                        # scaler.step(optimizer)
                        # scaler.update()
                        # lr_scheduler.step()
                        # optimizer.zero_grad()

                        scaler.scale(loss).backward()

                        if args.use_grad_clip:
                            # Unscales the gradients of optimizer's assigned params in-place
                            scaler.unscale_(optimizer)
                            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1e6)

                        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                        scaler.step(optimizer)

                        # Updates the scale for next iteration.
                        scaler.update()

                        lr_scheduler.step()
                        optimizer.zero_grad()

                        # When not using scaler
                        # loss.backward()
                        # torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                        # optimizer.step()
                        # lr_scheduler.step()
                        # optimizer.zero_grad()
                # else:
                #     with unet.join(), text_encoder.join():
                #         # Predict the noise residual and compute loss
                #         with torch.autocast('cuda', enabled=args.fp16):
                #             noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                            
                #         loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                #         # backprop and update
                #         scaler.scale(loss).backward()
                #         torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                #         torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), 1.0)
                #         scaler.step(optimizer)
                #         scaler.update()
                #         lr_scheduler.step()
                #         optimizer.zero_grad()                    

                # Update EMA
                if args.use_ema:
                    ema_unet.step(unet.parameters())

                # perf
                b_end = time.perf_counter()
                seconds_per_step = b_end - b_start
                steps_per_second = 1 / seconds_per_step
                rank_images_per_second = args.batch_size * steps_per_second
                world_images_per_second = rank_images_per_second * world_size
                samples_seen = global_step * args.batch_size * world_size

                # get global loss for logging
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                loss = loss / world_size

                if rank == 0:
                    progress_bar.update(1)
                    global_step += 1
                    logs = {
                        "train/loss": loss.detach().item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "train/samples_seen": samples_seen,
                        "perf/rank_samples_per_second": rank_images_per_second,
                        "perf/global_samples_per_second": world_images_per_second,
                    }
                    progress_bar.set_postfix(logs)
                    run.log(logs, step=global_step)

                if global_step % args.save_steps == 0 and global_step > 0:
                    save_checkpoint(global_step)
                if args.enableinference:
                    if global_step % args.image_log_steps == 0 and global_step > 0:
                        if rank == 0:
                            # get prompt from random batch
                            prompt = '' #tokenizer.decode(batch['tokens'][random.randint(0, len(batch['tokens'])-1)])

                            if args.image_log_scheduler == 'DDIMScheduler':
                                print('using DDIMScheduler scheduler')
                                scheduler = DDIMScheduler.from_pretrained(args.model, subfolder="scheduler", use_auth_token=args.hf_token)
                            else:
                                print('using PNDMScheduler scheduler')
                                scheduler=PNDMScheduler.from_pretrained(args.model, subfolder="scheduler", use_auth_token=args.hf_token)

                            pipeline = StableDiffusionPipeline(
                                text_encoder=text_encoder if type(text_encoder) is not torch.nn.parallel.DistributedDataParallel else text_encoder.module,
                                vae=vae,
                                unet=unet.module,
                                tokenizer=tokenizer,
                                scheduler=scheduler,
                                safety_checker=None, # disable safety checker to save memory
                                feature_extractor=None,#CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
                            ).to(device)
                            # inference
                            if args.enablewandb:
                                images = []
                            else:
                                saveInferencePath = args.output_path + "/inference"
                                os.makedirs(saveInferencePath, exist_ok=True)
                            with torch.no_grad():
                                with torch.autocast('cuda', enabled=args.fp16):
                                    for _ in range(args.image_log_amount):
                                        if args.enablewandb:
                                            images.append(
                                                wandb.Image(pipeline(
                                                    prompt, num_inference_steps=args.image_log_inference_steps
                                                ).images[0],
                                                caption=prompt)
                                            )
                                        else:
                                            from datetime import datetime
                                            images = pipeline(prompt, num_inference_steps=args.image_log_inference_steps,width=768,height=1024).images[0]
                                            filenameImg = str(time.time_ns()) + ".jpg"
                                            filenameTxt = str(time.time_ns()) + ".txt"
                                            images.save(saveInferencePath + "/" + filenameImg)
                                            with open(saveInferencePath + "/" + filenameTxt, 'a') as f:
                                                f.write('Used prompt: ' + prompt + '\n')
                                                f.write('Generated Image Filename: ' + filenameImg + '\n')
                                                f.write('Generated at: ' + str(global_step) + ' steps' + '\n')
                                                f.write('Generated at: ' + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))+ '\n')

                            # log images under single caption
                            if args.enablewandb:
                                run.log({'images': images}, step=global_step)

                            # cleanup so we don't run out of memory
                            del pipeline
                            gc.collect()
    except Exception as e:
        print(f'Exception caught on rank {rank} at step {global_step}, saving checkpoint...\n{e}\n{traceback.format_exc()}')
        pass

    save_checkpoint(global_step)

    cleanup()

    print(get_gpu_ram())
    print('Done!')

if __name__ == "__main__":
    setup()
    main()
