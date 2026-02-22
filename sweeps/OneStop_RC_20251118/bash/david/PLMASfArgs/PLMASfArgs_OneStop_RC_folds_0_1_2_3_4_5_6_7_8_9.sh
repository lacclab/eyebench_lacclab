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
    session_name="wandb-cpu${GPU_NUM}-dup${i}-unified-r0hb42bl-10"
    tmux new-session -d -s "${session_name}" "source ~/.conda/envs/eyebench/etc/profile.d/mamba.sh; cd /mnt/mlshare/reich3/eyebench_private; conda activate ${CONDA_ENV}; wandb agent EyeRead/OneStop_RC_20251118/r0hb42bl; wandb agent EyeRead/OneStop_RC_20251118/5xkvgikr; wandb agent EyeRead/OneStop_RC_20251118/s3ned9lq; wandb agent EyeRead/OneStop_RC_20251118/ekuy85yf; wandb agent EyeRead/OneStop_RC_20251118/r98plzfq; wandb agent EyeRead/OneStop_RC_20251118/9r8r6vgl; wandb agent EyeRead/OneStop_RC_20251118/dmcc93oa; wandb agent EyeRead/OneStop_RC_20251118/xth6xeqi; wandb agent EyeRead/OneStop_RC_20251118/4exp7ry7; wandb agent EyeRead/OneStop_RC_20251118/5you1gy3"; tmux set-option -t "${session_name}" remain-on-exit on
    echo "Launched W&B agent for CPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
