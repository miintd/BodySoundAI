#!/bin/bash

mkdir -p logs_3

LLM_MODELS=("gemma2B" "phi" "mistral" "llama" "deepseek-moe" "qwen-moe")
# LLM_MODELS=("GPT2")
NUM_RUNS=3
BATCH_SIZE=${1:-4}

# Mapping hidden size theo từng model
get_llm_dim() {
    case "$1" in
        GPT2) echo 768 ;;
        gemma2B) echo 2304 ;;
        phi)     echo 3072 ;;
        mistral) echo 4096 ;;
        llama)   echo 4096 ;;
        deepseek-moe) echo 2048 ;;
        qwen-moe) echo 2048 ;;
    esac
}

# no audio label
for MODEL in "${LLM_MODELS[@]}"; do
    LLM_DIM=$(get_llm_dim "$MODEL")

    for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
        run_name="${MODEL}_run${RUN}"
        LOG_FILE="logs_3/out_${MODEL}_${RUN}.txt"
        
        if [ -f "$LOG_FILE" ]; then
            if grep -q "torch.cuda.OutOfMemoryError\|CUDA out of memory" "$LOG_FILE"; then
                echo "[OOM] Re-running $MODEL run $RUN"
                rm "$LOG_FILE"
            elif grep -q "Traceback" "$LOG_FILE"; then
                echo "[ERROR] Re-running $MODEL run $RUN (non-OOM error)"
                rm "$LOG_FILE"
            else
                echo "[SKIP] $MODEL run $RUN (assumed success)"
                continue
            fi
        fi
        echo "  Model: $MODEL | Run: $RUN / $NUM_RUNS | llm_dim: $LLM_DIM"

        python src/benchmark/RespLLM/RespLLM.py \
            --llm_model "$MODEL" \
            --train_tasks S1,S2,S3,S4,S5,S6,S7 \
            --test_tasks T1,T2,T3,T4,T5,T6 \
            --train_epochs 60 \
            --meta_val_interval 3 \
            --train_pct 1 \
            --batch_size "$BATCH_SIZE" \
            --llm_dim "$LLM_DIM" \
            --d_ff "$LLM_DIM" \
            --wandb_name "$run_name" \
            2>&1 | tee -a logs_3/out_${MODEL}_${RUN}.txt

        echo "  Done: $MODEL run $RUN"
    done
done

# with audio label
for MODEL in "${LLM_MODELS[@]}"; do
    LLM_DIM=$(get_llm_dim "$MODEL")

    for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
        run_name="${MODEL}_audio_label_run${RUN}"
        LOG_FILE="logs_3/out_${MODEL}_audiolabel_${RUN}.txt"
        
        if [ -f "$LOG_FILE" ]; then
            if grep -q "torch.cuda.OutOfMemoryError\|CUDA out of memory" "$LOG_FILE"; then
                echo "[OOM] Re-running $MODEL (audiolabel) run $RUN"
                rm "$LOG_FILE"
            elif grep -q "Traceback" "$LOG_FILE"; then
                echo "[ERROR] Re-running $MODEL (audiolabel) run $RUN (non-OOM error)"
                rm "$LOG_FILE"
            else
                echo "[SKIP] $MODEL (audiolabel) run $RUN (assumed success)"
                continue
            fi
        fi
        echo "  Model: $MODEL | Run: $RUN / $NUM_RUNS | llm_dim: $LLM_DIM"

        python src/benchmark/RespLLM/RespLLM.py \
            --llm_model "$MODEL" \
            --train_tasks S1,S2,S3,S4,S5,S6,S7 \
            --test_tasks T1,T2,T3,T4,T5,T6 \
            --train_epochs 60 \
            --meta_val_interval 3 \
            --train_pct 1 \
            --batch_size "$BATCH_SIZE" \
            --llm_dim "$LLM_DIM" \
            --d_ff "$LLM_DIM" \
            --use_audiolabel \
            --wandb_name "$run_name" \
            2>&1 | tee -a logs_3/out_${MODEL}_audiolabel_${RUN}.txt

        echo "  Done: $MODEL (audiolabel) run $RUN"
    done
done