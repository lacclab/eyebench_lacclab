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
    session_name="wandb-cpu${GPU_NUM}-dup${i}-unified-369y2yte-10"
    tmux new-session -d -s "${session_name}" "source ~/.conda/envs/eyebench/etc/profile.d/mamba.sh; cd /mnt/mlshare/reich3/eyebench_private; conda activate ${CONDA_ENV}; wandb agent EyeRead/OneStop_RC_20251118/369y2yte; wandb agent EyeRead/OneStop_RC_20251118/na141z2l; wandb agent EyeRead/OneStop_RC_20251118/1qi2g7cc; wandb agent EyeRead/OneStop_RC_20251118/4jqw4a3r; wandb agent EyeRead/OneStop_RC_20251118/o1av5aa7; wandb agent EyeRead/OneStop_RC_20251118/6z9hy9e7; wandb agent EyeRead/OneStop_RC_20251118/qvlnri4h; wandb agent EyeRead/OneStop_RC_20251118/84yi6rmu; wandb agent EyeRead/OneStop_RC_20251118/4tkmkhp5; wandb agent EyeRead/OneStop_RC_20251118/eagm37a1"; tmux set-option -t "${session_name}" remain-on-exit on
    echo "Launched W&B agent for CPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
