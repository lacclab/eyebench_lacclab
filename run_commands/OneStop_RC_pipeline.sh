#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

DATA_TASK="OneStop_RC"
DL_PROJECT="OneStop_RC_20251118"
ML_PROJECT="OneStop_RC_20251104"

declare -A JOB_TOKEN
declare -A JOB_EXIT_FILE
declare -A JOB_RUNNER

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

usage() {
    cat <<EOF
Usage:
  bash run_commands/OneStop_RC_pipeline.sh [stage ...]

Stages:
  1_data_prep          Download and preprocess OneStop dataset
  4_sweeps             Generate sweep configurations for DL and ML models
  5_train              Run all model trainings (ML bash + DL sbatch)
  6_eval               Synchronize outputs and run final evaluation
  all                  Execute all stages in order

Examples:
  bash run_commands/OneStop_RC_pipeline.sh all
  bash run_commands/OneStop_RC_pipeline.sh 1_data_prep 4_sweeps
  bash run_commands/OneStop_RC_pipeline.sh 5_train 6_eval

Note:
  - Stage numbers follow the original runbook order
  - Stages 2 (model_checker) and 3 (sync_cleanup) are skipped per requirements
EOF
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing required command: $1" >&2
        exit 1
    }
}

start_tmux_job() {
    local session="$1"
    shift
    local cmd="$*"
    local token="eyebench_${session}_done"
    local exit_file="/tmp/${session}_$$.exit"
    local runner="/tmp/${session}_$$.sh"

    if tmux has-session -t "$session" 2>/dev/null; then
        log "Killing existing tmux session: $session"
        tmux kill-session -t "$session"
    fi

    cat >"$runner" <<EOF
#!/usr/bin/env bash
set +e
cd "$REPO_ROOT"
$cmd
code=\$?
echo "\$code" > "$exit_file"
tmux wait-for -S "$token"
exit \$code
EOF
    chmod +x "$runner"

    tmux new-session -d -s "$session" "$runner"
    JOB_TOKEN["$session"]="$token"
    JOB_EXIT_FILE["$session"]="$exit_file"
    JOB_RUNNER["$session"]="$runner"
    log "Started tmux session: $session"
}

wait_tmux_job() {
    local session="$1"
    local token="${JOB_TOKEN[$session]:-}"
    local exit_file="${JOB_EXIT_FILE[$session]:-}"
    local runner="${JOB_RUNNER[$session]:-}"

    [[ -n "$token" ]] || {
        echo "Unknown tmux session tracking key: $session" >&2
        exit 1
    }

    tmux wait-for "$token"
    local code=1
    if [[ -f "$exit_file" ]]; then
        code="$(cat "$exit_file")"
    fi

    rm -f "$exit_file" "$runner"

    if tmux has-session -t "$session" 2>/dev/null; then
        tmux kill-session -t "$session" || true
    fi

    if [[ "$code" != "0" ]]; then
        echo "tmux session '$session' failed with exit code $code" >&2
        exit "$code"
    fi
    log "Completed tmux session: $session"
}

# ============================================================================
# Stage 1: Data Preparation
# ============================================================================
#
# Downloads and preprocesses the OneStop dataset.
#
# Actions:
#   - Fetches raw data from configured sources
#   - Applies preprocessing pipeline (tokenization, feature extraction, etc.)
#   - Creates 10-fold cross-validation splits with strict subject/item separation
#   - Generates dataset statistics and feature summaries
#
# Output:
#   - data/OneStop/Raw/               (original data files)
#   - data/OneStop/Processed/         (preprocessed data)
#   - data/OneStop/folds_metadata/    (fold definitions and statistics)
#   - data/OneStop/folds/             (fold train/val/test splits)
#
# Time estimate: 30–60 minutes (network speed + preprocessing complexity)
# Dependencies: None (initial stage)

stage_1_data_prep() {
    log "====================================================================="
    log "Stage 1: Data Preparation"
    log "====================================================================="
    log "Downloading and preprocessing OneStop dataset..."
    log "Progress will be logged to stdout."
    log ""
    
    start_tmux_job data_onestop "bash src/data/preprocessing/get_data.sh OneStop"
    wait_tmux_job data_onestop
    
    log ""
    log "✓ Data preparation complete."
    log "  Dataset ready in: data/OneStop/"
    log "  Folds available in: data/OneStop/folds/"
}

# ============================================================================
# Stage 4: Sweep Configuration Generation
# ============================================================================
#
# Creates hyperparameter sweep configurations for all models.
#
# Actions:
#   - DL models: Generates SLURM job files + WandB sweep configs for all models
#   - ML models: Generates bash runner scripts for local execution
#   - Test wrapper: Creates unified test script for DL model evaluation
#
# Models included:
#   - DL: BEyeLSTM, AhnRNN, AhnCNN, MAG, PLMAS, PLMASf, RoberteyeWord, 
#         RoberteyeFixation, Roberta, PostFusion (variations: normal+basic features)
#   - ML: DummyClassifier, SVM, LogisticRegression, LogisticMeziere, RandomForest
#
# Generated outputs:
#   - sweeps/${DL_PROJECT}/slurm/*/              (SLURM job files per model)
#   - sweeps/${DL_PROJECT}/test_dl_wrapper.sh    (DL evaluation runner)
#   - sweeps/${ML_PROJECT}/bash/lacc/*/          (bash scripts per ML model)
#
# Time estimate: 2–5 minutes
# Dependencies: [Requires: 1_data_prep]

stage_4_sweeps() {
    log "====================================================================="
    log "Stage 4: Sweep Configuration Generation"
    log "====================================================================="
    log "Generating hyperparameter sweeps for DL models (WandB format)..."
    bash run_commands/utils/sweep_wrapper.sh \
        --data_tasks "$DATA_TASK" \
        --folds 0,1,2,3,4,5,6,7,8,9 \
        --wandb_project "$DL_PROJECT"
    
    log "Generating test wrapper script for DL model evaluation..."
    bash run_commands/utils/test_wrapper_creator.sh \
        --data_task "$DATA_TASK" \
        --project_name "$DL_PROJECT"
    
    log ""
    log "✓ Sweep configurations generated."
    log "  DL SLURM jobs: sweeps/${DL_PROJECT}/slurm/"
    log "  DL test wrapper: sweeps/${DL_PROJECT}/test_dl_wrapper.sh"
    log "  ML bash scripts: sweeps/${ML_PROJECT}/bash/lacc/"
}

# ============================================================================
# Stage 5: Model Training
# ============================================================================
#
# Trains all models across all 10 folds.
#
# Actions:
#   - ML models: Run locally via bash scripts with CPU
#   - DL models: Submit SLURM jobs for cluster execution (non-blocking)
#
# ML Training:
#   - Models: DummyClassifier, SVM, Logistic Regression, LogisticMeziere, 
#             Random Forest
#   - Execution: Sequential bash scripts on local CPU
#   - Expected time: 1–3 hours per model
#
# DL Training (via SLURM):
#   - 10 models × 2 feature variants (normal + basic) = 20 job configs
#   - Models: BEyeLSTM, AhnRNN, AhnCNN, MAG, PLMAS, PLMASf, RoberteyeWord,
#             RoberteyeFixation, Roberta, PostFusion
#   - Execution: SLURM queue (non-blocking sbatch submissions)
#   - Expected time: 24–48 hours total (depends on GPU queue availability)
#
# Results stored in:
#   - results/raw/{model_path}/fold_index_{i}/       (checkpoint + metrics)
#   - logs/wandb/                                      (WandB run tracking)
#
# Environment:
#   - CUDA devices: Automatically allocated by SLURM
#
# Time estimate: 1–3 hours (ML) + 24–48 hours (DL on cluster queue)
# Dependencies: [Requires: 4_sweeps]

stage_5_train() {
    log "====================================================================="
    log "Stage 5: Model Training"
    log "====================================================================="
    
    log "Running ML model training (bash scripts, CPU-based)..."
    log "Models: DummyClassifier, SVM, LogisticRegression, LogisticMeziere, RandomForest"
    log "This may take 1–3 hours."
    log ""
    
    bash "sweeps/${ML_PROJECT}/bash/lacc/DummyClassifierMLArgs/DummyClassifierMLArgs_${DATA_TASK}_folds_0_1_2_3_4_5_6_7_8_9.sh" || {
        log "❌ DummyClassifier training failed"; exit 1
    }
    
    bash "sweeps/${ML_PROJECT}/bash/lacc/SupportVectorMachineMLArgs/SupportVectorMachineMLArgs_${DATA_TASK}_folds_0_1_2_3_4_5_6_7_8_9.sh" || {
        log "❌ SVM training failed"; exit 1
    }
    
    bash "sweeps/${ML_PROJECT}/bash/lacc/LogisticRegressionMLArgs/LogisticRegressionMLArgs_${DATA_TASK}_folds_0_1_2_3_4_5_6_7_8_9.sh" || {
        log "❌ LogisticRegression training failed"; exit 1
    }
    
    bash "sweeps/${ML_PROJECT}/bash/lacc/LogisticMeziereArgs/LogisticMeziereArgs_${DATA_TASK}_folds_0_1_2_3_4_5_6_7_8_9.sh" || {
        log "❌ LogisticMeziere training failed"; exit 1
    }
    
    bash "sweeps/${ML_PROJECT}/bash/lacc/RandomForestMLArgs/RandomForestMLArgs_${DATA_TASK}_folds_0_1_2_3_4_5_6_7_8_9.sh" || {
        log "❌ RandomForest training failed"; exit 1
    }
    
    log ""
    log "✓ ML training complete. Checkpoints in: results/raw/"
    log ""
    log "Note: DL SLURM jobs are not submitted in this simplified pipeline."
    log "To train DL models, manually submit SLURM jobs from:"
    log "  sweeps/${DL_PROJECT}/slurm/"
}

# ============================================================================
# Stage 6: Evaluation & Results Processing
# ============================================================================
#
# Runs final model evaluation on held-out test sets.
#
# Actions:
#   - DL evaluation: Loads trained DL checkpoints, computes test-set metrics
#   - ML evaluation: Loads trained ML models, computes test-set metrics
#
# Metrics computed:
#   - Per-fold classification accuracy, F1, precision, recall
#   - Aggregated results across all 10 folds
#
# Output locations:
#   - results/raw/{model_name}/trial_level_test_results.csv   (per-fold results)
#   - results/processed/                                        (aggregated results)
#
# Generated files can be post-processed into paper tables via:
#   python src/run/multi_run/csv_to_latex.py
#
# Time estimate: 2–6 hours (depends on model count and DL inference speed)
# Dependencies: [Requires: 5_train, 4_sweeps]

stage_6_eval() {
    log "====================================================================="
    log "Stage 6: Evaluation & Results Processing"
    log "====================================================================="
    
    log "Running DL model evaluation (loading checkpoints, computing test metrics)..."
    log "This may take 2–4 hours."
    log ""
    start_tmux_job eval_onestop_rc_dl "CUDA_VISIBLE_DEVICES=0 bash sweeps/${DL_PROJECT}/test_dl_wrapper.sh"
    
    log "Running ML model evaluation (computing test metrics)..."
    log "This may take 1–2 hours."
    log ""
    start_tmux_job eval_onestop_rc_ml "python src/run/single_run/test_ml.py --data_task ${DATA_TASK} --wandb_project ${ML_PROJECT}"
    
    wait_tmux_job eval_onestop_rc_dl
    wait_tmux_job eval_onestop_rc_ml
    
    log ""
    log "✓ Evaluation complete."
    log "  Results available in: results/raw/ and results/processed/"
    log ""
    log "Next steps:"
    log "  1. Inspect trial_level_test_results.csv for per-fold metrics"
    log "  2. Run: python src/run/multi_run/csv_to_latex.py"
    log "     to generate benchmark tables for publication"
}

main() {
    require_cmd tmux
    require_cmd bash

    local stages=("$@")
    if [[ ${#stages[@]} -eq 0 ]]; then
        stages=(all)
    fi

    for stage in "${stages[@]}"; do
        case "$stage" in
        1_data_prep) stage_1_data_prep ;;
        4_sweeps) stage_4_sweeps ;;
        5_train) stage_5_train ;;
        6_eval) stage_6_eval ;;
        all)
            stage_1_data_prep
            stage_4_sweeps
            stage_5_train
            stage_6_eval
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown stage: $stage" >&2
            usage
            exit 1
            ;;
        esac
    done

    log ""
    log "✓✓✓ Pipeline finished successfully."
}

main "$@"
