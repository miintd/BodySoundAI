MODE=${1:-full}
BATCH_SIZE=${2:-16}
HF_TOKEN=${3:-""}
WANDB_KEY=${4:-""}
LLM_MODEL=${5:-""}
SAVE_PTH_MULTI=${6:-""}    # Multimodal + rule-based
SAVE_PTH_CONTEXT=${7:-""}  # Context-only
SAVE_PTH_AUDIO=${8:-""}    # Audio-only

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
echo "Running in $MODE mode | Train: $TRAIN_TASKS | Test: $TEST_TASKS | Logs: $LOG_DIR"

DEFAULT_LLM_MODELS=("GPT2Medium" "GPT2Large" "GPT2XL" "gemma2B")
LLM_MODELS=(${LLM_MODEL:-"${DEFAULT_LLM_MODELS[@]}"})
NUM_RUNS=1

TEST_MODES=("balanced" "gr1v3" "gr2v3" "gr1v4" "gr2v4")

get_llm_dim() {
    case "$1" in
        GPT2|DistilGPT2|GPTNeo125M) echo 768 ;;
        GPTNeo1.3B)   echo 2048 ;;
        GPT2Medium)   echo 1024 ;;
        GPT2Large)    echo 1280 ;;
        GPT2XL)       echo 1600 ;;
        gemma2B)      echo 2304 ;;
        phi)          echo 3072 ;;
        mistral|llama|llama3|OpenBioLLM) echo 4096 ;;
        deepseek-moe|qwen-moe) echo 2048 ;;
    esac
}

# Usage: run_exp <log_prefix> <extra_flags> <save_pth> <model> <run> [test_mode]
# If test_mode is empty → full_dataset mode
run_exp() {
    local PREFIX=$1 EXTRA_FLAGS=$2 SAVE_PTH=$3 MODEL=$4 RUN=$5 TEST=${6:-""}
    local LLM_DIM=$(get_llm_dim "$MODEL")
    local CKPT_NAME=$(basename "$(dirname "$SAVE_PTH")")

    local LOG_FILE RUN_NAME EXTRA_ARGS
    if [ -n "$TEST" ]; then
        LOG_FILE="$LOG_DIR/${PREFIX}_${MODEL}_${RUN}_${TEST}_${CKPT_NAME}.txt"
        RUN_NAME="${MODEL}_${RUN}_${TEST}"
        EXTRA_ARGS="--test_mode $TEST"
    else
        LOG_FILE="$LOG_DIR/${PREFIX}_${MODEL}_${RUN}_full_${CKPT_NAME}.txt"
        RUN_NAME="${MODEL}_run${RUN}_full"
        EXTRA_ARGS="--full_dataset True"
    fi

    if [ -f "$LOG_FILE" ]; then
        if grep -q "torch.cuda.OutOfMemoryError\|CUDA out of memory" "$LOG_FILE"; then
            echo "[OOM]   Re-running $MODEL run $RUN ${TEST:-full}"
            rm "$LOG_FILE"
        elif grep -q "Traceback\|Error" "$LOG_FILE"; then
            echo "[ERROR] Re-running $MODEL run $RUN ${TEST:-full}"
            rm "$LOG_FILE"
        else
            echo "[SKIP]  $MODEL run $RUN ${TEST:-full}"
            return
        fi
    fi

    echo "  [$PREFIX] Model: $MODEL | Run: $RUN | llm_dim: $LLM_DIM | ${TEST:-full}"
    python src/benchmark/RespLLM/RespLLM_ori.py \
        --llm_model "$MODEL" \
        --train_tasks "$TRAIN_TASKS" \
        --test_tasks "$TEST_TASKS" \
        --batch_size "$BATCH_SIZE" \
        --llm_dim "$LLM_DIM" \
        --d_ff "$LLM_DIM" \
        --wandb_name "$RUN_NAME" \
        --from_pretrain True \
        --save_pth "$SAVE_PTH" \
        $EXTRA_FLAGS \
        $EXTRA_ARGS \
        ${HF_TOKEN:+--hf_token "$HF_TOKEN"} \
        ${WANDB_KEY:+--wandb_key "$WANDB_KEY"} \
        2>&1 | tee -a "$LOG_FILE"
    echo "  Done: $MODEL run $RUN ${TEST:-full}"
}

for MODEL in "${LLM_MODELS[@]}"; do
    for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
        # Stratified test modes
        for TEST in "${TEST_MODES[@]}"; do
            run_exp "out"     ""                  "$SAVE_PTH_MULTI"    "$MODEL" "$RUN" "$TEST"
            run_exp "audio"   "--no-use_context_" "$SAVE_PTH_AUDIO"   "$MODEL" "$RUN" "$TEST"
            run_exp "context" "--no-use_audio"    "$SAVE_PTH_CONTEXT" "$MODEL" "$RUN" "$TEST"
        done
        # Full dataset
        run_exp "out"     ""                  "$SAVE_PTH_MULTI"    "$MODEL" "$RUN"
        run_exp "audio"   "--no-use_context_" "$SAVE_PTH_AUDIO"   "$MODEL" "$RUN"
        run_exp "context" "--no-use_audio"    "$SAVE_PTH_CONTEXT" "$MODEL" "$RUN"
    done
done