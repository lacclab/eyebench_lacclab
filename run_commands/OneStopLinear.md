### 1. Generate Sweep Configurations

```bash
DATASETS=(L2 L1L2)
TYPES=(REP ORD)
PREVS=(HUNT GATH ALL)
TASK_SUFFIXES=(MICH LEX MICH_R MICH_L TOE TOE_LR)

for dataset in "${DATASETS[@]}"; do
	for type in "${TYPES[@]}"; do
		for prev in "${PREVS[@]}"; do
			for task in "${TASK_SUFFIXES[@]}"; do
				data_task="OneStop_${dataset}_${type}_${prev}_${task}"
				project_name="${data_task}_linear_20251118"

				bash run_commands/utils/sweep_wrapper.sh \
					--data_tasks "${data_task}" \
					--folds 0,1,2,3,4,5,6,7,8,9 \
					--wandb_project "${project_name}" \
					--accelerator cpu \
					--cpu_count 1 \
					--regression true

				bash run_commands/utils/test_wrapper_creator.sh \
					--data_task "${data_task}" \
					--project_name "${project_name}"
			done
		done
	done
done
```

### 2. Training
```bash
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

DATASETS=(L2 L1L2)
TYPES=(REP ORD)
PREVS=(HUNT GATH ALL)
TASK_SUFFIXES=(MICH LEX MICH_R MICH_L TOE TOE_LR)

for dataset in "${DATASETS[@]}"; do
	for type in "${TYPES[@]}"; do
		for prev in "${PREVS[@]}"; do
			for task in "${TASK_SUFFIXES[@]}"; do
				data_task="OneStop_${dataset}_${type}_${prev}_${task}"
				project_name="${data_task}_linear_20251118"

				for model in "${MODELS[@]}"; do
					bash "sweeps/${project_name}/bash/lacc/${model}/${model}_${data_task}_folds_0_1_2_3_4_5_6_7_8_9.sh"
				done
			done
		done
	done
done
```

### 3. Post-Training Evaluation

```bash
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

DATASETS=(L2 L1L2)
TYPES=(REP ORD)
PREVS=(HUNT GATH ALL)
TASK_SUFFIXES=(MICH LEX MICH_R MICH_L TOE TOE_LR)

for dataset in "${DATASETS[@]}"; do
	for type in "${TYPES[@]}"; do
		for prev in "${PREVS[@]}"; do
			for task in "${TASK_SUFFIXES[@]}"; do
				data_task="OneStop_${dataset}_${type}_${prev}_${task}"
				project_name="${data_task}_linear_20251118"

				for model in "${MODELS[@]}"; do
					python src/run/single_run/test_ml.py \
						--data_task "${data_task}" \
						--wandb_project "${project_name}" \
						--model_name "${model}"
				done
			done
		done
	done
done
```