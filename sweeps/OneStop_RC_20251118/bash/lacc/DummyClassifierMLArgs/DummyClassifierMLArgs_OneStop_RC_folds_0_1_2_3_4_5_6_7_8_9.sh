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
    session_name="wandb-cpu${GPU_NUM}-dup${i}-unified-71gh84cz-10"
    tmux new-session -d -s "${session_name}" "source $HOME/miniforge3/etc/profile.d/conda.sh; cd $HOME/eyebench_private; conda activate ${CONDA_ENV}; wandb agent EyeRead/OneStop_RC_20251118/71gh84cz; wandb agent EyeRead/OneStop_RC_20251118/o07cit3c; wandb agent EyeRead/OneStop_RC_20251118/6hijpfn2; wandb agent EyeRead/OneStop_RC_20251118/dt7iw8tc; wandb agent EyeRead/OneStop_RC_20251118/ygc70bl9; wandb agent EyeRead/OneStop_RC_20251118/qffoc1sw; wandb agent EyeRead/OneStop_RC_20251118/k6abgelm; wandb agent EyeRead/OneStop_RC_20251118/ddmxpr8d; wandb agent EyeRead/OneStop_RC_20251118/g5xjz5kr; wandb agent EyeRead/OneStop_RC_20251118/v6si0s4v"; tmux set-option -t "${session_name}" remain-on-exit on
    echo "Launched W&B agent for CPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
