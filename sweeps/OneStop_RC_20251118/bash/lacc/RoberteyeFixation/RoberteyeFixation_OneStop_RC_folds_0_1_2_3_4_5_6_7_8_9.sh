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
    session_name="wandb-cpu${GPU_NUM}-dup${i}-unified-2p2mnqc7-10"
    tmux new-session -d -s "${session_name}" "source $HOME/miniforge3/etc/profile.d/conda.sh; cd $HOME/eyebench_private; conda activate ${CONDA_ENV}; wandb agent EyeRead/OneStop_RC_20251118/2p2mnqc7; wandb agent EyeRead/OneStop_RC_20251118/ebchyz7v; wandb agent EyeRead/OneStop_RC_20251118/y25cv9ng; wandb agent EyeRead/OneStop_RC_20251118/6tfqps7b; wandb agent EyeRead/OneStop_RC_20251118/5der57r7; wandb agent EyeRead/OneStop_RC_20251118/yzpz1glr; wandb agent EyeRead/OneStop_RC_20251118/dql3gg8z; wandb agent EyeRead/OneStop_RC_20251118/somjbf3a; wandb agent EyeRead/OneStop_RC_20251118/4gz4nswb; wandb agent EyeRead/OneStop_RC_20251118/eq8msrfh"; tmux set-option -t "${session_name}" remain-on-exit on
    echo "Launched W&B agent for CPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
