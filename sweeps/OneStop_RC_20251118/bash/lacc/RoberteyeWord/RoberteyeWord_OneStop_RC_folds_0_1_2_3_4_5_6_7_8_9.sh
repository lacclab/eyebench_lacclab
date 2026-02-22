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
    session_name="wandb-cpu${GPU_NUM}-dup${i}-unified-a1dgza08-10"
    tmux new-session -d -s "${session_name}" "source $HOME/miniforge3/etc/profile.d/conda.sh; cd $HOME/eyebench_private; conda activate ${CONDA_ENV}; wandb agent EyeRead/OneStop_RC_20251118/a1dgza08; wandb agent EyeRead/OneStop_RC_20251118/11bk8wtp; wandb agent EyeRead/OneStop_RC_20251118/g4wakxj9; wandb agent EyeRead/OneStop_RC_20251118/569axjmy; wandb agent EyeRead/OneStop_RC_20251118/e1m1glau; wandb agent EyeRead/OneStop_RC_20251118/3r31up5b; wandb agent EyeRead/OneStop_RC_20251118/hho8knsv; wandb agent EyeRead/OneStop_RC_20251118/ni44bl2z; wandb agent EyeRead/OneStop_RC_20251118/c59u2ajy; wandb agent EyeRead/OneStop_RC_20251118/tqc9pp4i"; tmux set-option -t "${session_name}" remain-on-exit on
    echo "Launched W&B agent for CPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
