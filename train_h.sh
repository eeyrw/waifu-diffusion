python main.py \
    -t \
    --base configs/stable-diffusion/fp16-no-ema-finetune.yaml \
    --gpus 0, \
    --scale_lr False \
    --num_nodes 1 \
    --logdir /root/autodl-tmp \
    --max_epochs 50