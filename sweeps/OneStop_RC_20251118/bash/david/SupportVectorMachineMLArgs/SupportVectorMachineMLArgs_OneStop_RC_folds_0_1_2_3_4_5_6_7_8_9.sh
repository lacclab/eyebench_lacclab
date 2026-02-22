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
    session_name="wandb-cpu${GPU_NUM}-dup${i}-unified-ypky4mo8-10"
    tmux new-session -d -s "${session_name}" "source ~/.conda/envs/eyebench/etc/profile.d/mamba.sh; cd /mnt/mlshare/reich3/eyebench_private; conda activate ${CONDA_ENV}; wandb agent EyeRead/OneStop_RC_20251118/ypky4mo8; wandb agent EyeRead/OneStop_RC_20251118/4j9pbru9; wandb agent EyeRead/OneStop_RC_20251118/2jth7bb6; wandb agent EyeRead/OneStop_RC_20251118/mco3mwap; wandb agent EyeRead/OneStop_RC_20251118/cyp3wi93; wandb agent EyeRead/OneStop_RC_20251118/1n3w47om; wandb agent EyeRead/OneStop_RC_20251118/pgv9f5uk; wandb agent EyeRead/OneStop_RC_20251118/qfnfl5hn; wandb agent EyeRead/OneStop_RC_20251118/46oqchca; wandb agent EyeRead/OneStop_RC_20251118/2hff7gxp"; tmux set-option -t "${session_name}" remain-on-exit on
    echo "Launched W&B agent for CPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
