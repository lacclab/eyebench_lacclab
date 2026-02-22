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
    session_name="wandb-cpu${GPU_NUM}-dup${i}-unified-elie1he9-10"
    tmux new-session -d -s "${session_name}" "source $HOME/miniforge3/etc/profile.d/conda.sh; cd $HOME/eyebench_private; conda activate ${CONDA_ENV}; wandb agent EyeRead/OneStop_RC_20251118/elie1he9; wandb agent EyeRead/OneStop_RC_20251118/sbrt90vu; wandb agent EyeRead/OneStop_RC_20251118/f3sj3eqw; wandb agent EyeRead/OneStop_RC_20251118/knejn39n; wandb agent EyeRead/OneStop_RC_20251118/t6ywcq0v; wandb agent EyeRead/OneStop_RC_20251118/6q2z7eyt; wandb agent EyeRead/OneStop_RC_20251118/f1cntn98; wandb agent EyeRead/OneStop_RC_20251118/3ik5p7om; wandb agent EyeRead/OneStop_RC_20251118/tj2anmzp; wandb agent EyeRead/OneStop_RC_20251118/zrnkqtol"; tmux set-option -t "${session_name}" remain-on-exit on
    echo "Launched W&B agent for CPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
