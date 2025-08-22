import argparse
import math
import os
import torch
import torch.multiprocessing as mp
from collections import OrderedDict, defaultdict, Counter
from OptAspectRatioBucketDataset import AspectBucket, AspectBucketSampler, ImageStore
import matplotlib.pyplot as plt

class DummyImageStore(ImageStore):
    """继承 ImageStore，但禁用真实图片读取"""
    def __init__(self, args, data_dir: str):
        super().__init__(args, data_dir)
    
    def get_image(self, ref):
        return ref

def worker(rank, num_replicas, args, num_epochs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    store = DummyImageStore(args, args.train_data_dir)
    
    bucket = AspectBucket(store, args.num_buckets, args.train_batch_size, 
                          args.bucket_side_min, args.bucket_side_max, 64,
                          args.bucket_mode, args.resolution * args.resolution,
                          args.multi_resolution, max_ratio=2.3)
    
    sampler = AspectBucketSampler(bucket=bucket, num_replicas=num_replicas, rank=rank, batch_size=args.train_batch_size)

    image_epoch_buckets = defaultdict(list)
    bucket_count_map = defaultdict(int)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for batch in sampler:
            for idx, bw, bh in batch:
                image_epoch_buckets[idx].append((bw, bh))
                bucket_count_map[(bw, bh)] += 1

    # 写入每个 rank 的文件，按图片编号排序
    rank_file = os.path.join(output_dir, f"image_bucket_rank{rank}.txt")
    with open(rank_file, "w") as f:
        f.write("ImageIdx\tEpochBuckets\n")
        for idx in sorted(image_epoch_buckets.keys()):
            buckets_str = ";".join([f"({bw},{bh})" for bw, bh in image_epoch_buckets[idx]])
            f.write(f"{idx}\t{buckets_str}\n")

    # 每个 rank 的桶使用次数
    bucket_file = os.path.join(output_dir, f"bucket_usage_rank{rank}.txt")
    with open(bucket_file, "w") as f:
        f.write("BucketWidth\tBucketHeight\tUsageCount\n")
        for (bw, bh), count in bucket_count_map.items():
            f.write(f"{bw}\t{bh}\t{count}\n")

    print(f"[Rank {rank}] done. {len(image_epoch_buckets)} images, {len(bucket_count_map)} buckets used.")

def merge_rank_outputs(output_dir, num_replicas,
                       merged_file="merged_image_bucket.txt",
                       merged_bucket_file="merged_bucket_usage.txt",
                       stats_file="bucket_stats.txt"):
    """整合 rank 输出并生成统计分析"""
    merged_image_map = defaultdict(list)
    merged_bucket_count = defaultdict(int)

    for rank in range(num_replicas):
        # 图片桶
        rank_file = os.path.join(output_dir, f"image_bucket_rank{rank}.txt")
        with open(rank_file, "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                idx, buckets_str = line.strip().split("\t")
                idx = int(idx)
                buckets = [tuple(map(int, b.strip("()").split(","))) for b in buckets_str.split(";")]
                merged_image_map[idx].extend(buckets)
        # 桶计数
        bucket_file = os.path.join(output_dir, f"bucket_usage_rank{rank}.txt")
        with open(bucket_file, "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                bw, bh, count = line.strip().split("\t")
                merged_bucket_count[(int(bw), int(bh))] += int(count)

    # 写入整合文件
    merged_file_path = os.path.join(output_dir, merged_file)
    with open(merged_file_path, "w") as f:
        f.write("ImageIdx\tAllEpochBuckets\n")
        for idx in sorted(merged_image_map.keys()):
            buckets_str = ";".join([f"({bw},{bh})" for bw, bh in merged_image_map[idx]])
            f.write(f"{idx}\t{buckets_str}\n")

    merged_bucket_file_path = os.path.join(output_dir, merged_bucket_file)
    with open(merged_bucket_file_path, "w") as f:
        f.write("BucketWidth\tBucketHeight\tTotalUsageCount\n")
        for (bw, bh), count in OrderedDict(sorted(merged_bucket_count.items(), key=lambda item: item[1], reverse=True)).items():
            f.write(f"{bw}\t{bh}\t{count}\n")
    # ----------------------
    # 每张图片训练次数直方图
    # ----------------------
    train_counts = [len(buckets) for buckets in merged_image_map.values()]
    counter = Counter(train_counts)
    xs = sorted(counter.keys())
    ys = [counter[x] for x in xs]
    plt.figure(figsize=(10,6))
    bars = plt.bar(xs, ys, width=0.8)
    plt.bar_label(bars, fmt="%d", label_type="edge", fontsize=8, padding=2)
    plt.xlabel("Number of times each image was used in training")
    plt.ylabel("Number of images")
    plt.title("Histogram of image training counts (all epochs and buckets)")
    plt.xticks(xs)
    # y 轴设为对数刻度
    plt.yscale("log")
    plt.tight_layout()
    train_count_file = os.path.join(output_dir, "image_train_count_histogram.png")
    plt.savefig(train_count_file)
    plt.close()
    print(f"Image training count histogram saved to {train_count_file}")

    # ----------------------
    # 绘制按桶尺寸排序的直方图（带宽高标注）
    # ----------------------
    bucket_sizes = [(bw, bh, math.sqrt(bw*bh), count) 
                    for (bw, bh), count in merged_bucket_count.items()]
    bucket_sizes.sort(key=lambda x: x[2])  # 按 sqrt(w*h) 排序

    sizes = [x[2] for x in bucket_sizes]
    counts = [x[3] for x in bucket_sizes]
    labels = [f"{bw}×{bh}" for bw, bh, _, _ in bucket_sizes]  # 显示宽高

    plt.figure(figsize=(max(12, len(labels)//2),6))  # 宽度随桶数量增加
    plt.bar(range(len(sizes)), counts, tick_label=labels)
    plt.xlabel("Bucket (Width×Height)")
    plt.ylabel("Number of images")
    plt.title("Histogram of images per bucket sorted by bucket size")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    hist_file = os.path.join(output_dir, "bucket_histogram.png")
    plt.savefig(hist_file)
    plt.close()
    print(f"Bucket histogram with sizes saved to {hist_file}")

    # ======== 统计分析 ========
    total_images = len(merged_image_map)
    total_buckets = len(merged_bucket_count)

    # 每个桶出现次数分布
    bucket_usage_values = list(merged_bucket_count.values())
    max_usage = max(bucket_usage_values) if bucket_usage_values else 0
    min_usage = min(bucket_usage_values) if bucket_usage_values else 0
    avg_usage = sum(bucket_usage_values)/len(bucket_usage_values) if bucket_usage_values else 0

    # 每张图片的桶多样性统计
    image_bucket_diversity = [len(set(b_list)) for b_list in merged_image_map.values()]
    max_diversity = max(image_bucket_diversity) if image_bucket_diversity else 0
    min_diversity = min(image_bucket_diversity) if image_bucket_diversity else 0
    avg_diversity = sum(image_bucket_diversity)/len(image_bucket_diversity) if image_bucket_diversity else 0

    # 写统计分析文件
    stats_file_path = os.path.join(output_dir, stats_file)
    with open(stats_file_path, "w") as f:
        f.write(f"TotalImages\t{total_images}\n")
        f.write(f"TotalUniqueBuckets\t{total_buckets}\n")
        f.write(f"BucketUsage_Max\t{max_usage}\n")
        f.write(f"BucketUsage_Min\t{min_usage}\n")
        f.write(f"BucketUsage_Avg\t{avg_usage:.2f}\n")
        f.write(f"ImageBucketDiversity_Max\t{max_diversity}\n")
        f.write(f"ImageBucketDiversity_Min\t{min_diversity}\n")
        f.write(f"ImageBucketDiversity_Avg\t{avg_diversity:.2f}\n")

    print(f"Merged outputs written: {merged_file_path}, {merged_bucket_file_path}, stats: {stats_file_path}")

def run_distributed_test(args, num_replicas=4, num_epochs=10, output_dir="bucket_test_output"):
    mp.spawn(worker, args=(num_replicas, args, num_epochs, output_dir), nprocs=num_replicas, join=True)
    merge_rank_outputs(output_dir, num_replicas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default='xxx',
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--multi_resolution",
        type=lambda x:list(map(int, x.split(','))),
        default=[512,574,640,704,768,832,896,960,1024],
        help=(
            'The multiple resolution bucket'
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    def bool_t(x): return x.lower() in ['true', 'yes', '1']
    parser.add_argument('--num_buckets', type=int, default=16,
                        help='The number of buckets.')
    parser.add_argument('--bucket_mode', type=str, default='maxfit',
                        help='multiscale|maxfit')
    parser.add_argument('--bucket_side_min', type=int, default=256,
                        help='The minimum side length of a bucket.')
    parser.add_argument('--bucket_side_max', type=int, default=1280,
                        help='The maximum side length of a bucket.')
    parser.add_argument('--ucg', type=float, default=0.1,
                        help='Percentage chance of dropping out the text condition per batch. Ranges from 0.0 to 1.0 where 1.0 means 100% text condition dropout.')  # 10% dropout probability
    parser.add_argument('--shuffle', dest='shuffle', type=bool_t,
                        default='True', help='Shuffle dataset')
    parser.add_argument('--output_bucket_info', type=bool_t,
                        default='False', help='Outputs bucket information and exits')
    parser.add_argument('--resize', type=bool_t, default='True',
                        help="Resizes dataset's images to the appropriate bucket dimensions.")
    parser.add_argument('--extended_validation', type=bool_t, default='False',
                        help='Perform extended validation of images to catch truncated or corrupt images.')
    parser.add_argument('--no_migration', type=bool_t, default='True',
                        help='Do not perform migration of dataset while the `--resize` flag is active. Migration creates an adjacent folder to the dataset with <dataset_dirname>_cropped.')
    parser.add_argument('--skip_validation', type=bool_t, default='True',
                        help='Skip validation of images, useful for speeding up loading of very large datasets that have already been validated.')

    parser.add_argument('--clip_penultimate', type=bool_t, default='False',
                        help='Use penultimate CLIP layer for text embedding')
    parser.add_argument('--extended_mode_chunks', type=int, default=0,
                        help='Enables extended mode for tokenization with given amount of maximum chunks. Values < 2 disable.')
    parser.add_argument('--local_files_only', type=bool_t, default='False',
                        help='Do not connect to HF')
    parser.add_argument('--weighted_sample', type=bool_t, default='False',
                        help='Use weighted sample')
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    args = parser.parse_args()

    run_distributed_test(args, num_replicas=4, output_dir='arbTestoutput')
