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
    session_name="wandb-cpu${GPU_NUM}-dup${i}-unified-8sam4x9i-10"
    tmux new-session -d -s "${session_name}" "source $HOME/miniforge3/etc/profile.d/conda.sh; cd $HOME/eyebench_private; conda activate ${CONDA_ENV}; wandb agent EyeRead/OneStop_RC_20251118/8sam4x9i; wandb agent EyeRead/OneStop_RC_20251118/qw43y5gr; wandb agent EyeRead/OneStop_RC_20251118/ivr8f05b; wandb agent EyeRead/OneStop_RC_20251118/hrq2u7u6; wandb agent EyeRead/OneStop_RC_20251118/n1a8570q; wandb agent EyeRead/OneStop_RC_20251118/4jdggnuh; wandb agent EyeRead/OneStop_RC_20251118/vxdknpik; wandb agent EyeRead/OneStop_RC_20251118/kk4lpdz8; wandb agent EyeRead/OneStop_RC_20251118/k6zbwv0r; wandb agent EyeRead/OneStop_RC_20251118/ifuk2ai7"; tmux set-option -t "${session_name}" remain-on-exit on
    echo "Launched W&B agent for CPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
