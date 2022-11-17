if [ ! -d "/dev/shm/FinalDs" ]; then
  cp -r /root/autodl-tmp/FinalDs /dev/shm/
fi
python main.py \
    -t \
    --base configs/stable-diffusion/minimal-ram-single-gpu.yaml \
    --gpus 0, \
    --scale_lr False \
    --num_nodes 1 \
    --logdir /root/autodl-tmp \
    --max_epochs 50