if [ ! -d "/dev/shm/FinalDs" ]; then
  cp -r /root/autodl-tmp/FinalDs /dev/shm/
fi
python main.py \
    -t \
    --base configs/stable-diffusion/fp16-no-ema-finetune.yaml \
    --gpus 0,1 \
    --scale_lr False \
    --num_nodes 1 \
    --logdir /root/autodl-tmp \
    --max_epochs 50