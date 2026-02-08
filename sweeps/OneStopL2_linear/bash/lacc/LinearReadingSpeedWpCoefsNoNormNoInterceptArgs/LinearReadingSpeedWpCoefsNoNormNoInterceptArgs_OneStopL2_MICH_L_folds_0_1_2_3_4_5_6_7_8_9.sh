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
    session_name="wandb-gpu${GPU_NUM}-dup${i}-unified-dc0y0rqk-10"
    tmux new-session -d -s "${session_name}" "conda activate eyebench; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/dc0y0rqk; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/6ckjeyiv; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/huls2rri; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/cre73jt6; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/o27obxgx; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/2p6218u3; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/7srp1jv6; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/360m90vm; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/78cafbek; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/ts6wexjq"; tmux set-option -t "${session_name}" remain-on-exit off
    echo "Launched W&B agent for GPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
