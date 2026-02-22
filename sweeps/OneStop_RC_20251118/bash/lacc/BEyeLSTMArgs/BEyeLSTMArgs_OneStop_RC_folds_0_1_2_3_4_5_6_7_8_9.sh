#!/bin/bash

if ! command -v tmux &>/dev/null; then
    echo "tmux could not be found, please install tmux first."
    exit 1  
fi

source $HOME/miniforge3/etc/profile.d/conda.sh
cd $HOME/eyebench_private

CONDA_ENV=${CONDA_ENV:-/data/home/ido.falah/miniforge3/envs/prof_env}
GPU_NUM=$1
RUNS_ON_GPU=${2:-1}
for ((i=1; i<=RUNS_ON_GPU; i++)); do
    session_name="wandb-cpu${GPU_NUM}-dup${i}-unified-8spz9xf5-10"
    tmux new-session -d -s "${session_name}" "source $HOME/miniforge3/etc/profile.d/conda.sh; cd $HOME/eyebench_private; conda activate ${CONDA_ENV}; wandb agent EyeRead/OneStop_RC_20251118/8spz9xf5; wandb agent EyeRead/OneStop_RC_20251118/2qnhwvj3; wandb agent EyeRead/OneStop_RC_20251118/4wb87zqb; wandb agent EyeRead/OneStop_RC_20251118/v56skt4u; wandb agent EyeRead/OneStop_RC_20251118/1vwqg4k5; wandb agent EyeRead/OneStop_RC_20251118/0oo1zewd; wandb agent EyeRead/OneStop_RC_20251118/kr86d2if; wandb agent EyeRead/OneStop_RC_20251118/ywmcfixa; wandb agent EyeRead/OneStop_RC_20251118/eh9zr03t; wandb agent EyeRead/OneStop_RC_20251118/k3l9tft0"; tmux set-option -t "${session_name}" remain-on-exit on
    echo "Launched W&B agent for CPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
