"""下载 HuggingFace 数据集到本地，支持离线加载。

Usage::

    # 下载到本地目录
    python download_hf_dataset.py --dataset_name TAAC2026/data_sample_1000 --save_dir ./data_sample_1000

    # 下载完整数据集（TAAC2026/data_sample_1000 -> TAAC2026/full_data）
    python download_hf_dataset.py --dataset_name TAAC2026/full_data --save_dir ./data_full

    # 下载并切分 train/valid
    python download_hf_dataset.py --dataset_name TAAC2026/data_sample_1000 --save_dir ./data_sample_1000 --split valid_ratio:0.1

加载方式::

    # 在线加载（需要网络）
    ds = load_dataset("TAAC2026/data_sample_1000")

    # 离线加载（本地目录）
    ds = load_dataset("./data_sample_1000")
"""

import os
import argparse
from pathlib import Path

from datasets import load_dataset, DatasetDict


def parse_args():
    parser = argparse.ArgumentParser(description="Download HF dataset to local disk")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="HuggingFace dataset name or path")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Local directory to save the dataset")
    parser.add_argument("--split", type=str, default="",
                        help='Optional split config, e.g. "valid_ratio:0.1" to create train/valid')
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splitting")
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from HuggingFace: {args.dataset_name}")
    ds = load_dataset(args.dataset_name)
    print(f"Original dataset: {ds}")

    if args.split.startswith("valid_ratio:"):
        valid_ratio = float(args.split.split(":", 1)[1])
        train_hf = ds["train"]
        split = train_hf.train_test_split(test_size=valid_ratio, seed=args.seed)
        ds = DatasetDict({
            "train": split["train"],
            "valid": split["test"],
        })
        print(f"After split (valid_ratio={valid_ratio}):")
        for k, v in ds.items():
            print(f"  {k}: {len(v)} rows")

    # Save in Arrow format (default for datasets, fastest reload)
    ds.save_to_disk(str(save_dir))
    print(f"Dataset saved to: {save_dir.absolute()}")

    # Also export as Parquet for cross-language compatibility
    parquet_dir = save_dir / "parquet"
    parquet_dir.mkdir(exist_ok=True)
    for split_name, dataset in ds.items():
        parquet_path = parquet_dir / f"{split_name}.parquet"
        dataset.to_parquet(str(parquet_path))
        print(f"  Parquet exported: {parquet_path}")

    print("\n--- 离线加载示例 ---")
    print(f'from datasets import load_from_disk')
    print(f'ds = load_from_disk("{save_dir}")')
    print(f"# 或在 train_hf.py 中使用: --dataset_name {save_dir}")


if __name__ == "__main__":
    main()
