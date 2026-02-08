#!/bin/bash

if ! command -v tmux &>/dev/null; then
    echo "tmux could not be found, please install tmux first."
    exit 1  
fi

source $HOME/miniforge3/etc/profile.d/conda.sh
cd $HOME/eyebench_private

GPU_NUM=$1
RUNS_ON_GPU=${2:-1}
for ((i=1; i<=RUNS_ON_GPU; i++)); do
    session_name="wandb-gpu${GPU_NUM}-dup${i}-unified-540v28ui-10"
    tmux new-session -d -s "${session_name}" "conda activate eyebench; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/540v28ui; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/u58501u3; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/v2uunih9; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/m9uv5la7; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/5uidl5yj; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/d2nrppnp; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/vz0bt8lb; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/62pxl6a0; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/nn8k5ntr; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/4vvlbgne"; tmux set-option -t "${session_name}" remain-on-exit off
    echo "Launched W&B agent for GPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
