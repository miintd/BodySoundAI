#!/bin/bash
# Usage: bash scripts/run_llm_baseline.sh [MODE] [BATCH_SIZE] [HF_TOKEN] [WANDB_KEY] [LLM_MODEL]
# MODE: "full" (default) or "covid"
# Example: bash scripts/run_llm_baseline.sh full 2 "hf_xxxxx" "wandb_xxxxx"
# Example: bash scripts/run_llm_baseline.sh covid 2 "hf_xxxxx" "wandb_xxxxx" GPT2Medium

set -uo pipefail

MODE=${1:-full}
BATCH_SIZE=${2:-16}
HF_TOKEN=${3:-""}
WANDB_KEY=${4:-""}
LLM_MODEL=${5:-""}

if [ "$MODE" = "covid" ]; then
    TRAIN_TASKS="S1,S2,S3,S4"
    TEST_TASKS="T1,T2,T3,T4"
    LOG_DIR="logs_rule_based_covid_1"
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
if [ -n "$LLM_MODEL" ]; then
    LLM_MODELS=("$LLM_MODEL")
else
    LLM_MODELS=("${DEFAULT_LLM_MODELS[@]}")
fi

TEST_MODES=("balanced" "full" "gr1v3" "gr2v3" "gr1v4" "gr2v4")
CKPT_ROOT="cks/llm_ori"

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
        llama3) echo 4096 ;;
        OpenBioLLM) echo 4096 ;;
        deepseek-moe) echo 2048 ;;
        qwen-moe) echo 2048 ;;
    esac
}

# ---------------------------------------------------------------------------
run_with_log() {
    local log_file="$1"; shift

    if [ -f "$log_file" ]; then
        if grep -q "torch.cuda.OutOfMemoryError\|CUDA out of memory" "$log_file"; then
            echo "    [OOM] Re-running -> $log_file"
            rm "$log_file"
        elif grep -q "Traceback\|Error" "$log_file"; then
            echo "    [ERROR] Re-running -> $log_file"
            rm "$log_file"
        else
            echo "    [SKIP] (assumed success) -> $log_file"
            return 0
        fi
    fi

    "$@" 2>&1 | tee -a "$log_file"
}

# ---------------------------------------------------------------------------
train_once() {
    local model="$1" llm_dim="$2" log_file="$3" wandb_name="$4"; shift 4
    local extra_flags=("$@")

    echo "  [TRAIN] Model: $model | llm_dim: $llm_dim | extra: ${extra_flags[*]:-none}"

    run_with_log "$log_file" python src/benchmark/RespLLM/RespLLM_ori.py \
        --llm_model "$model" \
        --train_tasks "$TRAIN_TASKS" \
        --test_tasks "$TEST_TASKS" \
        --batch_size "$BATCH_SIZE" \
        --llm_dim "$llm_dim" \
        --d_ff "$llm_dim" \
        --wandb_name "$wandb_name" \
        ${HF_TOKEN:+--hf_token "$HF_TOKEN"} \
        ${WANDB_KEY:+--wandb_key "$WANDB_KEY"} \
        "${extra_flags[@]}"
}

# ---------------------------------------------------------------------------
find_new_checkpoint() {
    local model="$1"
    local dir
    dir=$(find "$CKPT_ROOT" -maxdepth 1 -type d -name "${model}_*" -printf '%T@ %p\n' 2>/dev/null \
            | sort -rn | head -n1 | cut -d' ' -f2-)

    if [ -z "$dir" ]; then
        echo "ERROR: no checkpoint dir found for model $model under $CKPT_ROOT/" >&2
        return 1
    fi

    local ckpt="$dir/model_best.pt"
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: expected checkpoint not found at $ckpt" >&2
        return 1
    fi
    echo "$ckpt"
}

# ---------------------------------------------------------------------------
evaluate_checkpoint() {
    local model="$1" llm_dim="$2" ckpt="$3" test_mode="$4" log_file="$5" wandb_name="$6" use_rule_base="$7"; shift 7
    local extra_flags=("$@")

    local mode_flags=()
    if [ "$test_mode" = "full" ]; then
        mode_flags+=(--full_dataset True)
    else
        mode_flags+=(--test_mode "$test_mode")
    fi

    local rule_flags=()
    if [ "$use_rule_base" = "true" ]; then
        rule_flags+=(--use_rule_base True)
    fi

    echo "    [TEST] Model: $model | mode: $test_mode | rule_base: $use_rule_base | ckpt: $ckpt"

    run_with_log "$log_file" python src/benchmark/RespLLM/RespLLM_ori.py \
        --llm_model "$model" \
        --train_tasks "$TRAIN_TASKS" \
        --test_tasks "$TEST_TASKS" \
        --batch_size "$BATCH_SIZE" \
        --llm_dim "$llm_dim" \
        --d_ff "$llm_dim" \
        --from_pretrain True \
        --save_pth "$ckpt" \
        --wandb_name "$wandb_name" \
        ${HF_TOKEN:+--hf_token "$HF_TOKEN"} \
        ${WANDB_KEY:+--wandb_key "$WANDB_KEY"} \
        "${mode_flags[@]}" \
        "${rule_flags[@]}" \
        "${extra_flags[@]}"
}

# ---------------------------------------------------------------------------
run_configuration() {
    local config_name="$1" log_prefix="$2" run_rule_base="$3"; shift 3
    local extra_flags=("$@")

    for MODEL in "${LLM_MODELS[@]}"; do
        local llm_dim
        llm_dim=$(get_llm_dim "$MODEL")

        echo "===== [$config_name] $MODEL ====="

        local train_log="$LOG_DIR/${log_prefix}_train_${MODEL}.txt"
        train_once "$MODEL" "$llm_dim" "$train_log" "${MODEL}_${log_prefix}_train" "${extra_flags[@]}"

        local ckpt
        if ! ckpt=$(find_new_checkpoint "$MODEL"); then
            echo "  Skipping eval for $MODEL ($config_name): no checkpoint found"
            continue
        fi
        echo "  Found checkpoint: $ckpt"

        for TEST in "${TEST_MODES[@]}"; do
            local eval_log="$LOG_DIR/${log_prefix}_${TEST}_${MODEL}.txt"
            evaluate_checkpoint "$MODEL" "$llm_dim" "$ckpt" "$TEST" "$eval_log" \
                "${MODEL}_${log_prefix}_${TEST}" "false" "${extra_flags[@]}"
        done

        if [ "$run_rule_base" = "true" ]; then
            for TEST in "${TEST_MODES[@]}"; do
                local rb_log="$LOG_DIR/rulebase_${TEST}_${MODEL}.txt"
                evaluate_checkpoint "$MODEL" "$llm_dim" "$ckpt" "$TEST" "$rb_log" \
                    "${MODEL}_rulebase_${TEST}" "true" "${extra_flags[@]}"
            done
        fi
    done
}

# ===========================================================
# MULTIMODAL  (train -> test on all modes -> rule-based on same ckpt)
# ===========================================================
run_configuration "multimodal" "multimodal" "true"

# ===========================================================
# CONTEXT-ONLY  (train -> test on all modes)
# ===========================================================
run_configuration "context_only" "context" "false" --no-use_audio

# ===========================================================
# AUDIO-ONLY  (train -> test on all modes)
# ===========================================================
run_configuration "audio_only" "audio" "false" --no-use_context_

echo "All configurations finished."