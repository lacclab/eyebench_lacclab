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
    session_name="wandb-cpu${GPU_NUM}-dup${i}-unified-dpu2y46g-10"
    tmux new-session -d -s "${session_name}" "source $HOME/miniforge3/etc/profile.d/conda.sh; cd $HOME/eyebench_private; conda activate ${CONDA_ENV}; wandb agent EyeRead/OneStop_RC_20251118/dpu2y46g; wandb agent EyeRead/OneStop_RC_20251118/z5d9ohne; wandb agent EyeRead/OneStop_RC_20251118/hksjqhpz; wandb agent EyeRead/OneStop_RC_20251118/67usrm3r; wandb agent EyeRead/OneStop_RC_20251118/p98pmpkp; wandb agent EyeRead/OneStop_RC_20251118/o28doqau; wandb agent EyeRead/OneStop_RC_20251118/ihhs8ifi; wandb agent EyeRead/OneStop_RC_20251118/gi7pzd7q; wandb agent EyeRead/OneStop_RC_20251118/en6ooeyb; wandb agent EyeRead/OneStop_RC_20251118/9kpb6wcf"; tmux set-option -t "${session_name}" remain-on-exit on
    echo "Launched W&B agent for CPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
