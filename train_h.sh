python main.py \
    -t \
    --base configs/stable-diffusion/v1-4-finetune-hunk.yaml \
    --gpus 0, \
    --scale_lr False \
    --num_nodes 1 \
    --check_val_every_n_epoch 20000 \
    --finetune_from /root/autodl-tmp/v1-5-pruned-no_ema.ckpt \
    --logdir /root/autodl-tmp