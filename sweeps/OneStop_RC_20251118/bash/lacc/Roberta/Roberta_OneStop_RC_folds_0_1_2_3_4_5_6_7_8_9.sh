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
    session_name="wandb-cpu${GPU_NUM}-dup${i}-unified-jkrdvoe7-10"
    tmux new-session -d -s "${session_name}" "source $HOME/miniforge3/etc/profile.d/conda.sh; cd $HOME/eyebench_private; conda activate ${CONDA_ENV}; wandb agent EyeRead/OneStop_RC_20251118/jkrdvoe7; wandb agent EyeRead/OneStop_RC_20251118/uzjvsr8e; wandb agent EyeRead/OneStop_RC_20251118/gn2hs2yy; wandb agent EyeRead/OneStop_RC_20251118/n83uhtcu; wandb agent EyeRead/OneStop_RC_20251118/gisl7ui7; wandb agent EyeRead/OneStop_RC_20251118/ca4mw851; wandb agent EyeRead/OneStop_RC_20251118/ukmvcnx4; wandb agent EyeRead/OneStop_RC_20251118/670tm6hm; wandb agent EyeRead/OneStop_RC_20251118/aawhp8wu; wandb agent EyeRead/OneStop_RC_20251118/ksiwrtgj"; tmux set-option -t "${session_name}" remain-on-exit on
    echo "Launched W&B agent for CPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
