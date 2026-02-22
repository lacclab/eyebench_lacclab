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
    session_name="wandb-cpu${GPU_NUM}-dup${i}-unified-dccrfihh-10"
    tmux new-session -d -s "${session_name}" "source $HOME/miniforge3/etc/profile.d/conda.sh; cd $HOME/eyebench_private; conda activate ${CONDA_ENV}; wandb agent EyeRead/OneStop_RC_20251118/dccrfihh; wandb agent EyeRead/OneStop_RC_20251118/m2zfaivz; wandb agent EyeRead/OneStop_RC_20251118/fszvwxtc; wandb agent EyeRead/OneStop_RC_20251118/sq0v73mn; wandb agent EyeRead/OneStop_RC_20251118/5zvt0n3r; wandb agent EyeRead/OneStop_RC_20251118/lhg8d9zd; wandb agent EyeRead/OneStop_RC_20251118/i9247k60; wandb agent EyeRead/OneStop_RC_20251118/g5febl1s; wandb agent EyeRead/OneStop_RC_20251118/xkrxznaa; wandb agent EyeRead/OneStop_RC_20251118/7vwqqmly"; tmux set-option -t "${session_name}" remain-on-exit on
    echo "Launched W&B agent for CPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
