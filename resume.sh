if [ ! -d "/dev/shm/FinalDsWds" ]; then
  cp -r /root/autodl-tmp/FinalDsWds /dev/shm/
fi
python main.py \
    -t \
    --base configs/stable-diffusion/fp16-no-ema-finetune.yaml \
    --gpus 0,1 \
    --scale_lr False \
    --num_nodes 1 \
    --logdir /root/autodl-tmp \
    --max_epochs 50 \
    --resume /root/autodl-tmp/2022-10-31T21-17-39_fp16-no-ema-finetune