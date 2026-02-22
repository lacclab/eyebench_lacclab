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
    session_name="wandb-cpu${GPU_NUM}-dup${i}-unified-jfcpr604-10"
    tmux new-session -d -s "${session_name}" "source $HOME/miniforge3/etc/profile.d/conda.sh; cd $HOME/eyebench_private; conda activate ${CONDA_ENV}; wandb agent EyeRead/OneStop_RC_20251118/jfcpr604; wandb agent EyeRead/OneStop_RC_20251118/4hgnvzgb; wandb agent EyeRead/OneStop_RC_20251118/e0yh1uhu; wandb agent EyeRead/OneStop_RC_20251118/dwhnlnzo; wandb agent EyeRead/OneStop_RC_20251118/8gjgacod; wandb agent EyeRead/OneStop_RC_20251118/hqurf0sl; wandb agent EyeRead/OneStop_RC_20251118/9ygvfg2q; wandb agent EyeRead/OneStop_RC_20251118/9x1ipf9k; wandb agent EyeRead/OneStop_RC_20251118/ashhwld1; wandb agent EyeRead/OneStop_RC_20251118/901xfh4w"; tmux set-option -t "${session_name}" remain-on-exit on
    echo "Launched W&B agent for CPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
