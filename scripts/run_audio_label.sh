#!/bin/bash

mkdir -p logs_2

LLM_MODELS=("gemma2B" "phi" "mistral" "llama" "deepseek-moe" "qwen-moe")
NUM_RUNS=3
BATCH_SIZE=${1:-16}
LR = ${2:-1e-4}

# Mapping hidden size theo từng model
get_llm_dim() {
    case "$1" in
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
        LOG_FILE="logs_2/out_${MODEL}_${RUN}.txt"
        
        if [ -f "$LOG_FILE" ]; then
            echo "  [SKIP] Log file $LOG_FILE exists. Skipping $MODEL run $RUN."
            continue
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
            --lr "$LR" \
            --llm_dim "$LLM_DIM" \
            --d_ff "$LLM_DIM" \
            --wandb_name "$run_name" \
            2>&1 | tee -a logs_2/out_${MODEL}_${RUN}.txt

        echo "  Done: $MODEL run $RUN"
    done
done

# with audio label
for MODEL in "${LLM_MODELS[@]}"; do
    LLM_DIM=$(get_llm_dim "$MODEL")

    for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
        run_name="${MODEL}_audio_label_run${RUN}"
        LOG_FILE="logs_2/out_${MODEL}_audiolabel_${RUN}.txt"
        
        if [ -f "$LOG_FILE" ]; then
            echo "  [SKIP] Log file $LOG_FILE exists. Skipping $MODEL (audiolabel) run $RUN."
            continue
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
            --lr "$LR" \
            --llm_dim "$LLM_DIM" \
            --d_ff "$LLM_DIM" \
            --use_audiolabel \
            --wandb_name "$run_name" \
            2>&1 | tee -a logs_2/out_${MODEL}_audiolabel_${RUN}.txt

        echo "  Done: $MODEL run $RUN"
    done
done