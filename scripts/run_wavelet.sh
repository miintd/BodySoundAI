#!/bin/bash
BATCH_SIZE=${1:-16}

set -euo pipefail

LOG_DIR="logs_2"
mkdir -p "$LOG_DIR"

PYTHON_CMD="python src/benchmark/RespLLM/RespLLM.py"

COMMON_ARGS=(
  --llm_model gemma2B
  --train_tasks S1,S2,S3,S4,S5,S6,S7
  --test_tasks T1,T2,T3,T4,T5,T6
  --train_epochs 40
  --meta_val_interval 3
  --train_pct 1
  --batch_size "$BATCH_SIZE"
  --threshold_mode "soft"
)

methods=("universal" "bayesshrink")

# -----------------------------
# 0) Baseline: no wavelet
# -----------------------------
run_name="baseline"
echo "Running $run_name"

$PYTHON_CMD \
  "${COMMON_ARGS[@]}" \
  --wandb_name "$run_name" \
  2>&1 | tee -a "$LOG_DIR/${run_name}.txt"


# -----------------------------
# 1) Ablation for cough
# -----------------------------
cough_wavelets=("db4" "sym4")
cough_levels=(4 6)

for w in "${cough_wavelets[@]}"
do
  for l in "${cough_levels[@]}"
  do
    for method in "${methods[@]}"
    do
      run_name="ablation_cough_${w}_L${l}_${method}"
      echo "Running $run_name"

      $PYTHON_CMD \
        "${COMMON_ARGS[@]}" \
        --wavelet "$w" \
        --wavelet_modality "cough" \
        --wavelet_level "$l" \
        --method "$method" \
        --wandb_name "$run_name" \
        2>&1 | tee -a "$LOG_DIR/${run_name}.txt"
    done
  done
done


# -----------------------------
# 2) Ablation for breath
# -----------------------------
breath_wavelets=("db8" "sym8")
breath_levels=(4 6)

for w in "${breath_wavelets[@]}"
do
  for l in "${breath_levels[@]}"
  do
    for method in "${methods[@]}"
    do
      run_name="ablation_breath_${w}_L${l}_${method}"
      echo "Running $run_name"

      $PYTHON_CMD \
        "${COMMON_ARGS[@]}" \
        --wavelet "$w" \
        --wavelet_modality "breath" \
        --wavelet_level "$l" \
        --method "$method" \
        --wandb_name "$run_name" \
        2>&1 | tee -a "$LOG_DIR/${run_name}.txt"
    done
  done
done


# -----------------------------
# 3) Ablation for lung
# -----------------------------
lung_wavelets=("coif2" "coif3")
lung_levels=(4 6)

for w in "${lung_wavelets[@]}"
do
  for l in "${lung_levels[@]}"
  do
    for method in "${methods[@]}"
    do
      run_name="ablation_lung_${w}_L${l}_${method}"
      echo "Running $run_name"

      $PYTHON_CMD \
        "${COMMON_ARGS[@]}" \
        --wavelet "$w" \
        --wavelet_modality "lung" \
        --wavelet_level "$l" \
        --method "$method" \
        --wandb_name "$run_name" \
        2>&1 | tee -a "$LOG_DIR/${run_name}.txt"
    done
  done
done


# -----------------------------
# 4) Ablation for all modalities
# -----------------------------
all_wavelets=("db4" "sym8" "coif3")
all_levels=(4 6)
method="bayesshrink"

for w in "${all_wavelets[@]}"
do
  for l in "${all_levels[@]}"
  do
      run_name="ablation_all_${w}_L${l}_${method}"
      echo "Running $run_name"

      $PYTHON_CMD \
        "${COMMON_ARGS[@]}" \
        --wavelet "$w" \
        --wavelet_modality "all" \
        --wavelet_level "$l" \
        --method "$method" \
        --wandb_name "$run_name" \
        2>&1 | tee -a "$LOG_DIR/${run_name}.txt"
  done
done