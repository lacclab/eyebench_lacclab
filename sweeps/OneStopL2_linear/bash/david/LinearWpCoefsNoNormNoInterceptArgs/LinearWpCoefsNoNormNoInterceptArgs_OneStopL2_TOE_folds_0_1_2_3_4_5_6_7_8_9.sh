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
    session_name="wandb-gpu${GPU_NUM}-dup${i}-unified-njz1k8es-10"
    tmux new-session -d -s "${session_name}" "conda activate eyebench; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/njz1k8es; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/hmco10xc; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/0shm4csj; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/m4ww22ya; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/pgm6ccs4; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/xz9nvpv6; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/1wfe8it9; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/on8z4hgh; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/h7aucia2; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/w8itjcwi"; tmux set-option -t "${session_name}" remain-on-exit off
    echo "Launched W&B agent for GPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
