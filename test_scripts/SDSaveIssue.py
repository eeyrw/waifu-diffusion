from types import MethodType
import torch
from diffusers import UNet2DConditionModel
from torch.distributed.optim import ZeroRedundancyOptimizer
import bitsandbytes as bnb

torch.distributed.init_process_group("nccl", init_method="env://")

rank = torch.distributed.get_rank()
torch.cuda.set_device(rank)

unet = UNet2DConditionModel.from_config({
    "_class_name": "UNet2DConditionModel",
    "_diffusers_version": "0.6.0",
    "act_fn": "silu",
    "attention_head_dim": 8,
    "block_out_channels": [
        320,
        640,
        1280,
        1280
    ],
    "center_input_sample": False,
    "cross_attention_dim": 768,
    "down_block_types": [
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D"
    ],
    "downsample_padding": 1,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,
    "layers_per_block": 2,
    "mid_block_scale_factor": 1,
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "out_channels": 4,
    "sample_size": 64,
    "up_block_types": [
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D"
    ]
}
)
        
unet.enable_gradient_checkpointing()
unet.set_use_memory_efficient_attention_xformers(True)

device = torch.device('cuda')
unet = unet.to(device, dtype=torch.float32)

unet = torch.nn.parallel.DistributedDataParallel(
    unet,
    device_ids=[rank],
    output_device=rank,
    gradient_as_bucket_view=True
)

optimizer_parameters = unet.parameters()    
optimizer = ZeroRedundancyOptimizer(
        optimizer_parameters,
        optimizer_class=bnb.optim.AdamW8bit,
        parameters_as_bucket_view=True,
        lr=1e-7,
        betas=(0.9, 0.9),
        eps=0.9,
        weight_decay=1e-6,
    )

import inspect
import itertools
import os
import re
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union
import safetensors
import torch
from torch import Tensor, device, nn

def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name

def save_pretrained(
    self,
    save_directory: Union[str, os.PathLike],
    is_main_process: bool = True,
    save_function: Callable = None,
    safe_serialization: bool = False,
    variant: Optional[str] = None,
):
    """
    Save a model and its configuration file to a directory so that it can be reloaded using the
    [`~models.ModelMixin.from_pretrained`] class method.
    Arguments:
        save_directory (`str` or `os.PathLike`):
            Directory to save a model and its configuration file to. Will be created if it doesn't exist.
        is_main_process (`bool`, *optional*, defaults to `True`):
            Whether the process calling this is the main process or not. Useful during distributed training and you
            need to call this function on all processes. In this case, set `is_main_process=True` only on the main
            process to avoid race conditions.
        save_function (`Callable`):
            The function to use to save the state dictionary. Useful during distributed training when you need to
            replace `torch.save` with another method. Can be configured with the environment variable
            `DIFFUSERS_SAVE_MODE`.
        safe_serialization (`bool`, *optional*, defaults to `False`):
            Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
        variant (`str`, *optional*):
            If specified, weights are saved in the format `pytorch_model.<variant>.bin`.
    """
    if os.path.isfile(save_directory):
        print(f"Provided path ({save_directory}) should be a directory, not a file")
        return
    os.makedirs(save_directory, exist_ok=True)
    model_to_save = self
    # Attach architecture to the config
    # Save the config
    if is_main_process:
        model_to_save.save_config(save_directory)
    # Save the model
    weights_name = "diffusion_pytorch_model.safetensors"
    weights_name = _add_variant(weights_name, variant)
    # Save the model
    safetensors.torch.save_model(
        model_to_save, os.path.join(save_directory, weights_name), metadata={"format": "pt"}
    )
    print(f"Model weights saved in {os.path.join(save_directory, weights_name)}")

# unet.module.save_pretrained = MethodType(save_pretrained, unet.module)

if rank == 0:
    with unet.join():
        unet.module.save_pretrained(
            f'test/unet')
    
torch.distributed.destroy_process_group()
