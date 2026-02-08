#!/bin/bash

if ! command -v tmux &>/dev/null; then
    echo "tmux could not be found, please install tmux first."
    exit 1  
fi

source ~/.conda/envs/eyebench/etc/profile.d/mamba.sh
cd /mnt/mlshare/reich3/eyebench_private

GPU_NUM=$1
RUNS_ON_GPU=${2:-1}
for ((i=1; i<=RUNS_ON_GPU; i++)); do
    session_name="wandb-gpu${GPU_NUM}-dup${i}-unified-z9vjjljx-10"
    tmux new-session -d -s "${session_name}" "conda activate eyebench; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/z9vjjljx; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/agcmfl7q; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/ly979psh; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/yfmf8pl3; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/mm9zosjj; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/tnzx4hhf; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/esuejv89; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/fj9qg4wn; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/c8mnxius; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/zhswbwx1"; tmux set-option -t "${session_name}" remain-on-exit off
    echo "Launched W&B agent for GPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
