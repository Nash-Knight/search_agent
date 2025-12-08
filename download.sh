#!/usr/bin/env bash

# pip install -U "huggingface_hub[cli]"

set -euo pipefail

if [ "$#" -lt 2 ]; then
	echo "Usage: $0 <model_id> <save_dir>"
	exit 2
fi

model_id="$1"
save_dir="$2"

# Extract model name from model_id (part after last /)
model_name="${model_id##*/}"
local_dir="$save_dir/$model_name"

mkdir -p "$local_dir"

export HF_ENDPOINT=https://hf-mirror.com

hf download "$model_id" --local-dir "$local_dir"

echo "Downloaded $model_id to $local_dir"
