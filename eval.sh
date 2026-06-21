#!/bin/bash
# Usage: bash scripts/run_llm_baseline.sh [MODE] [BATCH_SIZE] [HF_TOKEN] [WANDB_KEY]
# MODE: "full" (default) or "covid"
# Example: bash scripts/run_llm_baseline.sh full 2 "hf_xxxxx" "wandb_xxxxx"
# Example: bash scripts/run_llm_baseline.sh covid 2 "hf_xxxxx" "wandb_xxxxx"

MODE=${1:-full}
BATCH_SIZE=${2:-16}
HF_TOKEN=${3:-""}
WANDB_KEY=${4:-""}
SAVE_PTH=${5:-"cks/llm_ori/GPT2_20260603-191646/model_best.pt"}
LLM_MODEL=${6:-""}

# Set tasks and log directory based on mode
if [ "$MODE" = "covid" ]; then
    TRAIN_TASKS="S1,S2,S3,S4"
    TEST_TASKS="T1,T2,T3,T4"
    LOG_DIR="logs_rule_based_covid"
else
    TRAIN_TASKS="S1,S2,S3,S4,S5,S6,S7"
    TEST_TASKS="T1,T2,T3,T4,T5,T6"
    LOG_DIR="logs_rule_based"
fi

mkdir -p "$LOG_DIR"
echo "Running in $MODE mode"
echo "  Train tasks: $TRAIN_TASKS"
echo "  Test tasks: $TEST_TASKS"
echo "  Log directory: $LOG_DIR"

DEFAULT_LLM_MODELS=("GPT2Medium" "GPT2Large" "GPT2XL" "gemma2B")
# LLM_MODELS=("DistilGPT2" "GPT2" "GPTNeo125M" "GPTNeo1.3B" "GPT2Medium")
if [ -n "$LLM_MODEL" ]; then
    LLM_MODELS=("$LLM_MODEL")
else
    LLM_MODELS=("${DEFAULT_LLM_MODELS[@]}")
fi
NUM_RUNS=1

get_llm_dim() {
    case "$1" in
        GPT2) echo 768 ;;
        DistilGPT2) echo 768 ;;
        GPTNeo125M) echo 768 ;;
        GPTNeo1.3B) echo 2048 ;;
        GPT2Medium) echo 1024 ;;
        GPT2Large) echo 1280 ;;
        GPT2XL) echo 1600 ;;
        gemma2B) echo 2304 ;;
        phi)     echo 3072 ;;
        mistral) echo 4096 ;;
        llama)   echo 4096 ;;
        deepseek-moe) echo 2048 ;;
        qwen-moe) echo 2048 ;;
    esac
}

TEST_MODE=("balanced" "gr1v3" "gr2v3" "gr1v4" "gr2v4")
CKPT_NAME=$(basename "$(dirname "$SAVE_PTH")")

# ===========================================================
# MULTIMODAL
# ===========================================================
for MODEL in "${LLM_MODELS[@]}"; do
    LLM_DIM=$(get_llm_dim "$MODEL")
    for TEST in "${TEST_MODE[@]}"; do
    for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
        run_name="${MODEL}_${RUN}_${TEST}"
        LOG_FILE="$LOG_DIR/out_${MODEL}_${RUN}_${TEST}_${CKPT_NAME}.txt"
        
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
            --batch_size "$BATCH_SIZE" \
            --llm_dim "$LLM_DIM" \
            --d_ff "$LLM_DIM" \
            --test_mode "$TEST" \
            --wandb_name "$run_name" \
            ${HF_TOKEN:+--hf_token "$HF_TOKEN"} \
            ${WANDB_KEY:+--wandb_key "$WANDB_KEY"} \
            --from_pretrain True \
            --save_pth "$SAVE_PTH" \
            2>&1 | tee -a $LOG_FILE

        echo "  Done: $MODEL run $RUN"
    done
    done
done

# full 
for MODEL in "${LLM_MODELS[@]}"; do
    LLM_DIM=$(get_llm_dim "$MODEL")
    for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
        run_name="${MODEL}_run${RUN}_full"
        LOG_FILE="$LOG_DIR/out_${MODEL}_${RUN}_full_${CKPT_NAME}.txt"
        
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
            --train_tasks $TRAIN_TASKS \
            --test_tasks $TEST_TASKS \
            --batch_size "$BATCH_SIZE" \
            --llm_dim "$LLM_DIM" \
            --d_ff "$LLM_DIM" \
            --full_dataset True \
            --wandb_name "$run_name" \
            ${HF_TOKEN:+--hf_token "$HF_TOKEN"} \
            ${WANDB_KEY:+--wandb_key "$WANDB_KEY"} \
            --from_pretrain True \
            --save_pth "$SAVE_PTH" \
            2>&1 | tee -a $LOG_FILE

        echo "  Done: $MODEL run $RUN"
    done
done

# ===========================================================
# AUDIO-ONLY
# ===========================================================
for MODEL in "${LLM_MODELS[@]}"; do
    LLM_DIM=$(get_llm_dim "$MODEL")
    for TEST in "${TEST_MODE[@]}"; do
    for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
        run_name="${MODEL}_${RUN}_${TEST}"
        LOG_FILE="$LOG_DIR/audio_${MODEL}_${RUN}_${TEST}_${CKPT_NAME}.txt"
        
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
            --batch_size "$BATCH_SIZE" \
            --llm_dim "$LLM_DIM" \
            --d_ff "$LLM_DIM" \
            --no-use_context_ \
            --test_mode "$TEST" \
            --wandb_name "$run_name" \
            ${HF_TOKEN:+--hf_token "$HF_TOKEN"} \
            ${WANDB_KEY:+--wandb_key "$WANDB_KEY"} \
            --from_pretrain True \
            --save_pth "$SAVE_PTH" \
            2>&1 | tee -a $LOG_FILE

        echo "  Done: $MODEL run $RUN"
    done
    done
done

# full 
for MODEL in "${LLM_MODELS[@]}"; do
    LLM_DIM=$(get_llm_dim "$MODEL")
    for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
        run_name="${MODEL}_run${RUN}_full"
        LOG_FILE="$LOG_DIR/audio_${MODEL}_${RUN}_full_${CKPT_NAME}.txt"
        
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
            --train_tasks $TRAIN_TASKS \
            --batch_size "$BATCH_SIZE" \
            --llm_dim "$LLM_DIM" \
            --d_ff "$LLM_DIM" \
            --no-use_context_ \
            --full_dataset True \
            --wandb_name "$run_name" \
            ${HF_TOKEN:+--hf_token "$HF_TOKEN"} \
            ${WANDB_KEY:+--wandb_key "$WANDB_KEY"} \
            --from_pretrain True \
            --save_pth "$SAVE_PTH" \
            2>&1 | tee -a $LOG_FILE

        echo "  Done: $MODEL run $RUN"
    done
done

# ===========================================================
# CONTEXT-ONLY
# ===========================================================
for MODEL in "${LLM_MODELS[@]}"; do
    LLM_DIM=$(get_llm_dim "$MODEL")
    for TEST in "${TEST_MODE[@]}"; do
    for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
        run_name="${MODEL}_${RUN}_${TEST}"
        LOG_FILE="$LOG_DIR/context_${MODEL}_${RUN}_${TEST}_${CKPT_NAME}.txt"
        
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
            --batch_size "$BATCH_SIZE" \
            --llm_dim "$LLM_DIM" \
            --d_ff "$LLM_DIM" \
            --no-use_audio \
            --test_mode "$TEST" \
            --wandb_name "$run_name" \
            ${HF_TOKEN:+--hf_token "$HF_TOKEN"} \
            ${WANDB_KEY:+--wandb_key "$WANDB_KEY"} \
            --from_pretrain True \
            --save_pth "$SAVE_PTH" \
            2>&1 | tee -a $LOG_FILE

        echo "  Done: $MODEL run $RUN"
    done
    done
done

# full 
for MODEL in "${LLM_MODELS[@]}"; do
    LLM_DIM=$(get_llm_dim "$MODEL")
    for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
        run_name="${MODEL}_run${RUN}_full"
        LOG_FILE="$LOG_DIR/context_${MODEL}_${RUN}_full_${CKPT_NAME}.txt"
        
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
            --train_tasks $TRAIN_TASKS \
            --test_tasks $TEST_TASKS \
            --batch_size "$BATCH_SIZE" \
            --llm_dim "$LLM_DIM" \
            --d_ff "$LLM_DIM" \
            --no-use_audio \
            --full_dataset True \
            --wandb_name "$run_name" \
            ${HF_TOKEN:+--hf_token "$HF_TOKEN"} \
            ${WANDB_KEY:+--wandb_key "$WANDB_KEY"} \
            --from_pretrain True \
            --save_pth "$SAVE_PTH" \
            2>&1 | tee -a $LOG_FILE

        echo "  Done: $MODEL run $RUN"
    done
done

# ===========================================================
# RULE-BASED
# ===========================================================
for MODEL in "${LLM_MODELS[@]}"; do
    LLM_DIM=$(get_llm_dim "$MODEL")
    for TEST in "${TEST_MODE[@]}"; do
    for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
        run_name="${MODEL}_${RUN}_${TEST}"
        LOG_FILE="$LOG_DIR/rulebase_${MODEL}_${RUN}_${TEST}_${CKPT_NAME}.txt"
        
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
            --batch_size "$BATCH_SIZE" \
            --llm_dim "$LLM_DIM" \
            --d_ff "$LLM_DIM" \
            --use_rule_base True \
            --test_mode "$TEST" \
            --wandb_name "$run_name" \
            ${HF_TOKEN:+--hf_token "$HF_TOKEN"} \
            ${WANDB_KEY:+--wandb_key "$WANDB_KEY"} \
            --from_pretrain True \
            --save_pth "$SAVE_PTH" \
            2>&1 | tee -a $LOG_FILE

        echo "  Done: $MODEL run $RUN"
    done
    done
done

# full 
for MODEL in "${LLM_MODELS[@]}"; do
    LLM_DIM=$(get_llm_dim "$MODEL")
    for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
        run_name="${MODEL}_run${RUN}_full"
        LOG_FILE="$LOG_DIR/rulebase_${MODEL}_${RUN}_full_${CKPT_NAME}.txt"
        
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
            --train_tasks $TRAIN_TASKS \
            --test_tasks $TEST_TASKS \
            --batch_size "$BATCH_SIZE" \
            --llm_dim "$LLM_DIM" \
            --d_ff "$LLM_DIM" \
            --use_rule_base True \
            --full_dataset True \
            --wandb_name "$run_name" \
            ${HF_TOKEN:+--hf_token "$HF_TOKEN"} \
            ${WANDB_KEY:+--wandb_key "$WANDB_KEY"} \
            --from_pretrain True \
            --save_pth "$SAVE_PTH" \
            2>&1 | tee -a $LOG_FILE

        echo "  Done: $MODEL run $RUN"
    done
done