#!/bin/bash

mkdir -p logs_clap

encoder=("label")

# Default parameters
BATCH_SIZE=${1:-16}
EPOCHS=${2:-20}
LR=${3:-1e-4}
CLAP_FINAL=${4:-"concat"}
NUM_RUNS=${5:-1}

echo "=========================================="
echo "CLAP COVID Classification Training"
echo "=========================================="
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo "CLAP Final: $CLAP_FINAL"
echo "Num Runs: $NUM_RUNS"
echo "=========================================="

for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
    run_name="clap_run${RUN}"
    LOG_FILE="logs_clap/out_clap_${CLAP_FINAL}_run${RUN}_ori.txt"
    
    # Check if log file exists and contains errors
    if [ -f "$LOG_FILE" ]; then
        if grep -q "torch.cuda.OutOfMemoryError\|CUDA out of memory" "$LOG_FILE"; then
            echo "[OOM] Re-running CLAP run $RUN (out of memory)"
            rm "$LOG_FILE"
        elif grep -q "Traceback\|Error" "$LOG_FILE"; then
            echo "[ERROR] Re-running CLAP run $RUN (non-OOM error)"
            rm "$LOG_FILE"
        else
            echo "[SKIP] CLAP run $RUN (assumed success)"
            continue
        fi
    fi

    python src/benchmark/RespLLM/model_clap.py \
        --n_cls 2 \
        --train_tasks S1,S2,S3,S4 \
        --test_tasks T1,T2,T3,T4 \
        --batch_size "$BATCH_SIZE" \
        --train_epochs "$EPOCHS" \
        --lr "$LR" \
        --use_context True \
        --clap_model "laion/clap-htsat-unfused" \
        --clap_final "$CLAP_FINAL" \
        --test_interval 1 \
        --meta_val_interval 3 \
        --save_pth "cks/clap/clap_${CLAP_FINAL}_run${RUN}/model_best.pt" \
        2>&1 | tee "$LOG_FILE"
    
    echo "Completed run $RUN. Log saved to $LOG_FILE"
    echo ""
done

for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
for enc in "${encoder[@]}"; do
    LOG_FILE="logs_clap/out_clap_${CLAP_FINAL}_${enc}_run${RUN}.txt"

    if [ -f "$LOG_FILE" ]; then
        echo "  [SKIP] Log file $LOG_FILE exists. Skipping fusion run $RUN."
        continue
    fi

    echo "  Fusion Model | Run: $RUN / $NUM_RUNS | batch: 32"
    python src/benchmark/RespLLM/model_bts.py \
        --n_cls 2 \
        --train_tasks S1,S2,S3,S4 \
        --test_tasks T1,T2,T3,T4 \
        --batch_size "$BATCH_SIZE" \
        --train_epochs "$EPOCHS" \
        --lr "$LR" \
        --use_context True \
        --clap_model "laion/clap-htsat-unfused" \
        --clap_final "$CLAP_FINAL" \
        --test_interval 1 \
        --meta_val_interval 3 \
        --modality_encoder_type "$enc" \
        --modal_embs raw_concat \
        2>&1 | tee -a "$LOG_FILE"
done
done