#!/bin/bash
# Usage: bash scripts/run_llm_baseline.sh [MODE] [BATCH_SIZE] [HF_TOKEN] [WANDB_KEY]
# MODE: "full" (default) or "covid"
# Example: bash scripts/run_llm_baseline.sh full 2 "hf_xxxxx" "wandb_xxxxx"
# Example: bash scripts/run_llm_baseline.sh covid 2 "hf_xxxxx" "wandb_xxxxx"

MODE=${1:-full}
BATCH_SIZE=${2:-16}
HF_TOKEN=${3:-""}
WANDB_KEY=${4:-""}
LLM_MODEL=${5:-""}

# Set tasks and log directory based on mode
if [ "$MODE" = "covid" ]; then
    TRAIN_TASKS="S1,S2,S3,S4"
    TEST_TASKS="T1,T2,T3,T4"
    LOG_DIR="logs_llm_fix_seed/COVID"
else
    TRAIN_TASKS="S1,S2,S3,S4,S5,S6,S7"
    TEST_TASKS="T1,T2,T3,T4,T5,T6"
    LOG_DIR="logs_llm_fix_seed"
fi

mkdir -p "$LOG_DIR"
echo "Running in $MODE mode"
echo "  Train tasks: $TRAIN_TASKS"
echo "  Test tasks: $TEST_TASKS"
echo "  Log directory: $LOG_DIR"
if [ -z "$LLM_MODEL" ]; then
    echo "  LLM Models: Running all models in default list"
else
    echo "  LLM Model: $LLM_MODEL only"
fi

# LLM_MODELS=("gemma2B" "phi" "mistral" "llama" "deepseek-moe" "qwen-moe")
if [ -z "$LLM_MODEL" ]; then
    # Use default list if no model specified
    LLM_MODELS=("GPTNeo1.3B" "GPT2Medium" "gemma2B")
else
    # Use specified model
    LLM_MODELS=("$LLM_MODEL")
fi
# LLM_MODELS=("DistilGPT2" "GPT2" "GPTNeo125M" "GPTNeo1.3B" "GPT2Medium")
# LLM_MODELS=("GPT2")
NUM_RUNS=3

get_llm_dim() {
    case "$1" in
        GPT2) echo 768 ;;
        DistilGPT2) echo 768 ;;
        GPTNeo125M) echo 768 ;;
        GPTNeo1.3B) echo 2048 ;;
        GPT2Medium) echo 1024 ;;
        gemma2B) echo 2304 ;;
        phi)     echo 3072 ;;
        mistral) echo 4096 ;;
        llama)   echo 4096 ;;
        deepseek-moe) echo 2048 ;;
        qwen-moe) echo 2048 ;;
    esac
}


for MODEL in "${LLM_MODELS[@]}"; do
    LLM_DIM=$(get_llm_dim "$MODEL")
    for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
        run_name="${MODEL}_run${RUN}_${MODE}"
        LOG_FILE="$LOG_DIR/out_${MODEL}_${RUN}.txt"
        
        if [ -f "$LOG_FILE" ]; then
            if grep -q "torch.cuda.OutOfMemoryError\|CUDA out of memory" "$LOG_FILE"; then
                echo "[OOM] Re-running $MODEL run $RUN"
                rm "$LOG_FILE"
            elif grep -q "Traceback\|Error" "$LOG_FILE"; then
                echo "[ERROR] Re-running $MODEL run $RUN (non-OOM error)"
                rm "$LOG_FILE"
            else
                echo "[SKIP] $MODEL run $RUN (assumed success)"
                continue
            fi
        fi
        echo "  Model: $MODEL | Run: $RUN / $NUM_RUNS | llm_dim: $LLM_DIM"

        python src/benchmark/RespLLM/RespLLM_ori.py \
            --llm_model "$MODEL" \
            --train_tasks "$TRAIN_TASKS" \
            --test_tasks "$TEST_TASKS" \
            --train_epochs 60 \
            --meta_val_interval 3 \
            --train_pct 1 \
            --batch_size "$BATCH_SIZE" \
            --llm_dim "$LLM_DIM" \
            --d_ff "$LLM_DIM" \
            --wandb_name "$run_name" \
            --use_8bit_quantization \
            ${HF_TOKEN:+--hf_token "$HF_TOKEN"} \
            ${WANDB_KEY:+--wandb_key "$WANDB_KEY"} \
            2>&1 | tee -a $LOG_FILE

        echo "  Done: $MODEL run $RUN"
    done
done

# add val
# for MODEL in "${LLM_MODELS[@]}"; do
#     LLM_DIM=$(get_llm_dim "$MODEL")
#     for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
#         run_name="${MODEL}_run${RUN}_ori_add_val_S3_S4"
#         LOG_FILE="logs_COVID_tasks/out_${MODEL}_${RUN}_ori_add_val_S3_S4.txt"
        
#         if [ -f "$LOG_FILE" ]; then
#             if grep -q "torch.cuda.OutOfMemoryError\|CUDA out of memory" "$LOG_FILE"; then
#                 echo "[OOM] Re-running $MODEL run $RUN"
#                 rm "$LOG_FILE"
#             elif grep -q "Traceback\|Error" "$LOG_FILE"; then
#                 echo "[ERROR] Re-running $MODEL run $RUN (non-OOM error)"
#                 rm "$LOG_FILE"
#             else
#                 echo "[SKIP] $MODEL run $RUN (assumed success)"
#                 continue
#             fi
#         fi
#         echo "  Model: $MODEL | Run: $RUN / $NUM_RUNS | llm_dim: $LLM_DIM"

#         python src/benchmark/RespLLM/RespLLM_ori.py \
#             --llm_model "$MODEL" \
#             --train_tasks S1,S2,S3,S4 \
#             --test_tasks T1,T2,T3,T4 \
#             --train_epochs 60 \
#             --meta_val_interval 3 \
#             --train_pct 1 \
#             --batch_size "$BATCH_SIZE" \
#             --llm_dim "$LLM_DIM" \
#             --d_ff "$LLM_DIM" \
#             --no-val_S3_S4 \
#             --wandb_name "$run_name" \
#             2>&1 | tee -a $LOG_FILE

#         echo "  Done: $MODEL run $RUN"
#     done
# done

# context only
# for MODEL in "${LLM_MODELS[@]}"; do
#     LLM_DIM=$(get_llm_dim "$MODEL")
#     for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
#         run_name="${MODEL}_ori_run${RUN}_context"
#         LOG_FILE="logs_llm_fixed_valloss/out_${MODEL}_${RUN}_context.txt"
        
#         if [ -f "$LOG_FILE" ]; then
#             if grep -q "torch.cuda.OutOfMemoryError\|CUDA out of memory" "$LOG_FILE"; then
#                 echo "[OOM] Re-running $MODEL run $RUN"
#                 rm "$LOG_FILE"
#             elif grep -q "Traceback\|Error" "$LOG_FILE"; then
#                 echo "[ERROR] Re-running $MODEL run $RUN (non-OOM error)"
#                 rm "$LOG_FILE"
#             else
#                 echo "[SKIP] $MODEL run $RUN (assumed success)"
#                 continue
#             fi
#         fi
#         echo "  Model: $MODEL | Run: $RUN / $NUM_RUNS | llm_dim: $LLM_DIM"

#         python src/benchmark/RespLLM/RespLLM_ori.py \
#             --llm_model "$MODEL" \
#             --train_tasks S1,S2,S3,S4,S5,S6,S7 \
#             --test_tasks T1,T2,T3,T4,T5,T6 \
#             --train_epochs 60 \
#             --meta_val_interval 3 \
#             --train_pct 1 \
#             --batch_size "$BATCH_SIZE" \
#             --llm_dim "$LLM_DIM" \
#             --d_ff "$LLM_DIM" \
#             --no-use_audio \
#             --wandb_name "$run_name" \
#             --save_pth cks/llm_ori/out_${MODEL}_${RUN}_best_epoch.pt \
#             2>&1 | tee -a $LOG_FILE

#         echo "  Done: $MODEL run $RUN"
#     done
# done

# # context only, no S7
# for MODEL in "${LLM_MODELS[@]}"; do
#     LLM_DIM=$(get_llm_dim "$MODEL")
#     for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
#         run_name="${MODEL}_ori_run${RUN}_context_noS7"
#         LOG_FILE="logs_llm_fixed_valloss/out_${MODEL}_${RUN}_context_noS7.txt"
        
#         if [ -f "$LOG_FILE" ]; then
#             if grep -q "torch.cuda.OutOfMemoryError\|CUDA out of memory" "$LOG_FILE"; then
#                 echo "[OOM] Re-running $MODEL run $RUN"
#                 rm "$LOG_FILE"
#             elif grep -q "Traceback\|Error" "$LOG_FILE"; then
#                 echo "[ERROR] Re-running $MODEL run $RUN (non-OOM error)"
#                 rm "$LOG_FILE"
#             else
#                 echo "[SKIP] $MODEL run $RUN (assumed success)"
#                 continue
#             fi
#         fi
#         echo "  Model: $MODEL | Run: $RUN / $NUM_RUNS | llm_dim: $LLM_DIM"

#         python src/benchmark/RespLLM/RespLLM_ori.py \
#             --llm_model "$MODEL" \
#             --train_tasks S1,S2,S3,S4,S5,S6 \
#             --test_tasks T1,T2,T3,T4,T5,T6 \
#             --train_epochs 60 \
#             --meta_val_interval 3 \
#             --train_pct 1 \
#             --batch_size "$BATCH_SIZE" \
#             --llm_dim "$LLM_DIM" \
#             --d_ff "$LLM_DIM" \
#             --no-use_audio \
#             --wandb_name "$run_name" \
#             --save_pth cks/llm_ori/out_${MODEL}_${RUN}_best_epoch.pt \
#             2>&1 | tee -a $LOG_FILE

#         echo "  Done: $MODEL run $RUN"
#     done
# done