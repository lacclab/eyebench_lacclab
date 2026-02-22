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
    session_name="wandb-cpu${GPU_NUM}-dup${i}-unified-4ct4qpr2-10"
    tmux new-session -d -s "${session_name}" "source ~/.conda/envs/eyebench/etc/profile.d/mamba.sh; cd /mnt/mlshare/reich3/eyebench_private; conda activate ${CONDA_ENV}; wandb agent EyeRead/OneStop_RC_20251118/4ct4qpr2; wandb agent EyeRead/OneStop_RC_20251118/pnka3lrc; wandb agent EyeRead/OneStop_RC_20251118/5a25fi6i; wandb agent EyeRead/OneStop_RC_20251118/b2jji89r; wandb agent EyeRead/OneStop_RC_20251118/boe4qefs; wandb agent EyeRead/OneStop_RC_20251118/7em4ol6y; wandb agent EyeRead/OneStop_RC_20251118/1daylxkl; wandb agent EyeRead/OneStop_RC_20251118/q4ds1v3z; wandb agent EyeRead/OneStop_RC_20251118/iqyan8k6; wandb agent EyeRead/OneStop_RC_20251118/on5fgmjm"; tmux set-option -t "${session_name}" remain-on-exit on
    echo "Launched W&B agent for CPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
