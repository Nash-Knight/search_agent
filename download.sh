# pip install -U "huggingface_hub[cli]"

model_id=$1
save_dir=$2

# Extract model name from model_id (part after last /)
model_name=${model_id##*/}
local_dir="$save_dir/$model_name"

export HF_ENDPOINT=https://hf-mirror.com

hf download $model_id --local-dir $local_dir
