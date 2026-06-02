#!/bin/bash

mkdir -p logs_specific_modal

# LLM_MODELS=("gemma2B" "phi" "mistral" "llama" "deepseek-moe" "qwen-moe")
LLM_MODELS=("GPT2")
NUM_RUNS=3
BATCH_SIZE=${1:-16}
TRAIN_TASK=("S1,S3,S5" "S2,S4,S6" "S7")
TEST_TASK=("T3,T4" "T1,T2" "T5,T6")

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
    for TASK in "${TRAIN_TASK[@]}"; do
        for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
            run_name="${MODEL}_run${RUN}_${TASK}"
            LOG_FILE="logs_specific_modal/out_${MODEL}_${RUN}_${TASK}.txt"
            
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
                --train_tasks "$TASK" \
                --test_tasks "$TASK" \
                --train_epochs 40 \
                --meta_val_interval 3 \
                --train_pct 1 \
                --batch_size "$BATCH_SIZE" \
                --llm_dim "$LLM_DIM" \
                --d_ff "$LLM_DIM" \
                --wandb_name "$run_name" \
                2>&1 | tee -a "$LOG_FILE"

            echo "  Done: $MODEL run $RUN"
        done
    done
done

for MODEL in "${LLM_MODELS[@]}"; do
    LLM_DIM=$(get_llm_dim "$MODEL")

    for i in "${!TRAIN_TASK[@]}"; do
        TRAIN="${TRAIN_TASK[$i]}"
        TEST="${TEST_TASK[$i]}"

        for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
            run_name="${MODEL}_run${RUN}_${TEST}"
            LOG_FILE="logs_specific_modal/out_${MODEL}_${RUN}_${TEST}.txt"
            
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

            echo "Model: $MODEL | Run: $RUN | Train: $TRAIN | Test: $TEST"

            python src/benchmark/RespLLM/RespLLM.py \
                --llm_model "$MODEL" \
                --train_tasks "$TRAIN" \
                --test_tasks "$TEST" \
                --train_epochs 40 \
                --meta_val_interval 3 \
                --train_pct 1 \
                --batch_size "$BATCH_SIZE" \
                --llm_dim "$LLM_DIM" \
                --d_ff "$LLM_DIM" \
                --wandb_name "$run_name" \
                2>&1 | tee -a "$LOG_FILE"

            echo "Done: $MODEL run $RUN"
        done
    done
done
