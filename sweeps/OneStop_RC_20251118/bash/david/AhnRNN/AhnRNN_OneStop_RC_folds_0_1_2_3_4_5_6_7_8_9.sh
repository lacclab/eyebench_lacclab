#!/bin/bash

if ! command -v tmux &>/dev/null; then
    echo "tmux could not be found, please install tmux first."
    exit 1  
fi

source ~/.conda/envs/eyebench/etc/profile.d/mamba.sh
cd /mnt/mlshare/reich3/eyebench_private

CONDA_ENV=${CONDA_ENV:-eyebench}
GPU_NUM=$1
RUNS_ON_GPU=${2:-1}
for ((i=1; i<=RUNS_ON_GPU; i++)); do
    session_name="wandb-cpu${GPU_NUM}-dup${i}-unified-sq19i7c3-10"
    tmux new-session -d -s "${session_name}" "source ~/.conda/envs/eyebench/etc/profile.d/mamba.sh; cd /mnt/mlshare/reich3/eyebench_private; conda activate ${CONDA_ENV}; wandb agent EyeRead/OneStop_RC_20251118/sq19i7c3; wandb agent EyeRead/OneStop_RC_20251118/lcqhl2wq; wandb agent EyeRead/OneStop_RC_20251118/glto5myk; wandb agent EyeRead/OneStop_RC_20251118/j1ptbz12; wandb agent EyeRead/OneStop_RC_20251118/o5ohzhth; wandb agent EyeRead/OneStop_RC_20251118/j21nx9or; wandb agent EyeRead/OneStop_RC_20251118/fk1lqku5; wandb agent EyeRead/OneStop_RC_20251118/1s6evch6; wandb agent EyeRead/OneStop_RC_20251118/moai9osb; wandb agent EyeRead/OneStop_RC_20251118/byiq8pi2"; tmux set-option -t "${session_name}" remain-on-exit on
    echo "Launched W&B agent for CPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
