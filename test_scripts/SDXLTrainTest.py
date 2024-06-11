import torch
from diffusers import UNet2DConditionModel
from torch.distributed.optim import ZeroRedundancyOptimizer
import bitsandbytes as bnb

torch.distributed.init_process_group("nccl", init_method="env://")

rank = torch.distributed.get_rank()
torch.cuda.set_device(rank)

unet = UNet2DConditionModel.from_config({
    "_class_name": "UNet2DConditionModel",
    "_diffusers_version": "0.19.0.dev0",
    "act_fn": "silu",
    "addition_embed_type": "text_time",
    "addition_embed_type_num_heads": 64,
    "addition_time_embed_dim": 256,
    "attention_head_dim": [
        5,
        10,
        20
    ],
    "block_out_channels": [
        320,
        640,
        1280
    ],
    "center_input_sample": False,
    "class_embed_type": None,
    "class_embeddings_concat": False,
    "conv_in_kernel": 3,
    "conv_out_kernel": 3,
    "cross_attention_dim": 2048,
    "cross_attention_norm": None,
    "down_block_types": [
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D"
    ],
    "downsample_padding": 1,
    "dual_cross_attention": False,
    "encoder_hid_dim": None,
    "encoder_hid_dim_type": None,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,
    "layers_per_block": 2,
    "mid_block_only_cross_attention": None,
    "mid_block_scale_factor": 1,
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "num_attention_heads": None,
    "num_class_embeds": None,
    "only_cross_attention": False,
    "out_channels": 4,
    "projection_class_embeddings_input_dim": 2816,
    "resnet_out_scale_factor": 1.0,
    "resnet_skip_time_act": False,
    "resnet_time_scale_shift": "default",
    "sample_size": 128,
    "time_cond_proj_dim": None,
    "time_embedding_act_fn": None,
    "time_embedding_dim": None,
    "time_embedding_type": "positional",
    "timestep_post_act": None,
    "transformer_layers_per_block": [
        1,
        2,
        10
    ],
    "up_block_types": [
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D"
    ],
    "upcast_attention": None,
    "use_linear_projection": True
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

scaler = torch.cuda.amp.GradScaler(enabled=True)
for i in range(10):
    unet.train()
    latents = torch.randn(1, 64, 64)

    # Sample noise
    noise = torch.randn_like(latents)

    bsz = latents.shape[0]
    timesteps = torch.randint(0, 1000, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    noisy_latents = noisy_latents

    # Get the embedding for conditioning
    encoder_hidden_states = torch.randn(1, 77, 1024)
    target = torch.randn_like(latents)
    with unet.join():
        # Predict the noise residual and compute loss
        with torch.autocast('cuda', enabled=True):
            noise_pred = unet(noisy_latents, timesteps,
                              encoder_hidden_states).sample

        loss = torch.nn.functional.mse_loss(
            noise_pred.float(), target.float(), reduction="mean")

        # optimizer.zero_grad()

        scaler.scale(loss).backward()

        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        optimizer.zero_grad()

    # get global loss for logging
    torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)

torch.distributed.destroy_process_group()
