python main.py \
    -t \
    --base configs/autoencoder/autoencoder_simple.yaml \
    --gpus 0, \
    --scale_lr False \
    --num_nodes 1 \
    --logdir /root/autodl-tmp \
    --max_epochs 50