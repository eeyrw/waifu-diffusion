import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import UNet2DConditionModel

weight_dtype = torch.float16

cfg = {
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
  "cross_attention_dim": 2048,
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

unet = UNet2DConditionModel(**cfg)

unet.to('cuda', dtype=weight_dtype)
unet.set_use_memory_efficient_attention_xformers(True)
noise = torch.randn(1, 4, 64, 64).to('cuda', dtype=weight_dtype)
noisy_latents = torch.randn(1, 4, 64, 64).to('cuda', dtype=weight_dtype)
timesteps = torch.tensor(543, device='cuda', dtype=torch.int64)
encoder_hidden_states = torch.randn(1, 10, 2048).to('cuda', dtype=weight_dtype)
noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
loss.backward()