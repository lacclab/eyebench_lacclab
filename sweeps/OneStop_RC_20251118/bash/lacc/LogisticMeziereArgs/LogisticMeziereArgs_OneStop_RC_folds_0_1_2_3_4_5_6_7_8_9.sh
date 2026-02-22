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
    session_name="wandb-cpu${GPU_NUM}-dup${i}-unified-jbp20jtl-10"
    tmux new-session -d -s "${session_name}" "source $HOME/miniforge3/etc/profile.d/conda.sh; cd $HOME/eyebench_private; conda activate ${CONDA_ENV}; wandb agent EyeRead/OneStop_RC_20251118/jbp20jtl; wandb agent EyeRead/OneStop_RC_20251118/n5lsw2lg; wandb agent EyeRead/OneStop_RC_20251118/oj2dl2ac; wandb agent EyeRead/OneStop_RC_20251118/vxw8szyr; wandb agent EyeRead/OneStop_RC_20251118/hv1cnwze; wandb agent EyeRead/OneStop_RC_20251118/5k25joo2; wandb agent EyeRead/OneStop_RC_20251118/1zlw4yg6; wandb agent EyeRead/OneStop_RC_20251118/nlav5na0; wandb agent EyeRead/OneStop_RC_20251118/9swtunz9; wandb agent EyeRead/OneStop_RC_20251118/6dfmtxt5"; tmux set-option -t "${session_name}" remain-on-exit on
    echo "Launched W&B agent for CPU ${GPU_NUM}, Dup ${i} in tmux session ${session_name}"
done
