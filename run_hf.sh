#!/bin/bash
# run_hf.sh — PCVRHyFormer HF 数据集训练启动脚本 (Linux/macOS/Git Bash)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# 默认从 HuggingFace 在线加载；若已下载到本地，把 DATASET 改成本地路径即可
DATASET="${1:-TAAC2026/data_sample_1000}"
shift || true

python3 -u "${SCRIPT_DIR}/train_hf.py" \
    --dataset_name "${DATASET}" \
    --batch_size 32 \
    --lr 1e-4 \
    --num_epochs 10 \
    --d_model 64 \
    --emb_dim 64 \
    --num_queries 1 \
    --num_hyformer_blocks 2 \
    --num_heads 4 \
    --seq_encoder_type swiglu \
    --rank_mixer_mode ffn_only \
    --dropout_rate 0.01 \
    --patience 5 \
    --loss_type bce \
    --valid_ratio 0.1 \
    --emb_skip_threshold 1000000 \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --log_dir "${SCRIPT_DIR}/logs_hf" \
    --ckpt_dir "${SCRIPT_DIR}/checkpoints_hf" \
    "$@"
