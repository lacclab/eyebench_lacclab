#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_onestopl2_linear_single.sh
#
# This script trains a single ML model and then tests it (per model/task).
# It expects W&B to be available and uses the latest run ID from outputs.

PROJECT="EyeBench"
FOLDS=(0 1 2 3 4 5 6 7 8 9)

MODELS=(
  LinearRegressionArgs
  LinearFixationMetricsArgs
  LinearSClustersArgs
  LinearSClustersNoNormArgs
  LinearWpCoefsArgs
  LinearWpCoefsNoNormArgs
  LinearWpCoefsNoInterceptArgs
  LinearWpCoefsNoNormNoInterceptArgs
  LinearReadingSpeedFixationMetricsArgs
  LinearReadingSpeedSClustersArgs
  LinearReadingSpeedSClustersNoNormArgs
  LinearReadingSpeedWpCoefsArgs
  LinearReadingSpeedWpCoefsNoNormArgs
  LinearReadingSpeedWpCoefsNoInterceptArgs
  LinearReadingSpeedWpCoefsNoNormNoInterceptArgs
  LinearReadingSpeedFixationMetricsSClustersArgs
  LinearReadingSpeedFixationMetricsSClustersNoNormArgs
  LinearReadingSpeedFixationMetricsWpCoefsArgs
  LinearReadingSpeedFixationMetricsWpCoefsNoNormArgs
  LinearReadingSpeedFixationMetricsWpCoefsNoInterceptArgs
  LinearReadingSpeedFixationMetricsWpCoefsNoNormNoInterceptArgs
  LinearReadingSpeedSClustersWpCoefsArgs
  LinearReadingSpeedSClustersNoNormWpCoefsNoNormArgs
  LinearReadingSpeedSClustersWpCoefsNoInterceptArgs
  LinearReadingSpeedSClustersNoNormWpCoefsNoNormNoInterceptArgs
  LinearReadingSpeedFixationMetricsSClustersWpCoefsArgs
  LinearReadingSpeedFixationMetricsSClustersNoNormWpCoefsNoNormArgs
  LinearReadingSpeedFixationMetricsSClustersWpCoefsNoInterceptArgs
  LinearReadingSpeedFixationMetricsSClustersNoNormWpCoefsNoNormNoInterceptArgs
)

DATA_TASKS=(
  OneStopL2_MICH
  OneStopL2_LEX
  OneStopL2_MICH_R
  OneStopL2_MICH_L
  OneStopL2_TOE
  OneStopL2_TOE_LR

  OneStopL1
)

DATASETS=(L2 L1L2)
TYPES=(REP ORD)
PREVS=(HUNT GATH ALL)
TASK_SUFFIXES=(MICH LEX MICH_R MICH_L TOE TOE_LR)

for dataset in "${DATASETS[@]}"; do
  for type in "${TYPES[@]}"; do
    for prev in "${PREVS[@]}"; do
      for task in "${TASK_SUFFIXES[@]}"; do
        DATA_TASKS+=("OneStop_${dataset}_${type}_${prev}_${task}")
      done
    done
  done
done

for data in "${DATA_TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      echo "==== Training ${model} on ${data} (fold ${fold}) ===="
      python src/run/single_run/train.py \
        +trainer=TrainerML \
        +model="${model}" \
        +data="${data}" \
        data.fold_index=${fold}

      RUN_PATH=$(ls -td outputs/+data=${data},+model=${model},+trainer=TrainerML*/fold_index=${fold}/wandb/run-* 2>/dev/null | head -n 1 || true)
      if [[ -z "${RUN_PATH}" ]]; then
        echo "ERROR: Could not find W&B run directory for ${model} on ${data} (fold ${fold})."
        exit 1
      fi

      RUN_ID=$(basename "${RUN_PATH}" | sed -E 's/^run-[0-9_]+-([a-z0-9]+)$/\1/')
      if [[ -z "${RUN_ID}" ]]; then
        echo "ERROR: Could not extract W&B run ID from ${RUN_PATH}."
        exit 1
      fi

      echo "==== Testing ${model} on ${data} (fold ${fold}) with run_id=${RUN_ID} ===="
      python src/run/single_run/test_ml.py \
        --wandb_run_id "${RUN_ID}" \
        --data_task "${data}" \
        --wandb_project "${PROJECT}" \
        --model_name "${model}"
    done
  done
done
