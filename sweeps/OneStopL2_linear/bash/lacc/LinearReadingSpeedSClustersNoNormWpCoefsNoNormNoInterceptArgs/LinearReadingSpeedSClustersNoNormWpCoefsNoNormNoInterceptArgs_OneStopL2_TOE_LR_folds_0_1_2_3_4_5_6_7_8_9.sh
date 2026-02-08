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
    session_name="wandb-gpu${GPU_NUM}-dup${i}-unified-jc2f4w6c-10"
    tmux new-session -d -s "${session_name}" "conda activate eyebench; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/jc2f4w6c; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/vv35nkrk; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/9ksolpw2; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/4a9eudl4; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/5omfahiq; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/q3et8741; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/u8fsk7po; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/lrvw5m38; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/n64vak3n; CUDA_VISIBLE_DEVICES=${GPU_NUM} wandb agent EyeRead/OneStopL2_linear/meceucmg"; tmux set-option -t "${session_name}" remain-on-exit off
    echo "Launched W&B agent for GPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
