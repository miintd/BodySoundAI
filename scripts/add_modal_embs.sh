#!/bin/bash
# Usage: bash scripts/add_modal_embs.sh [MODE] [BATCH_SIZE] [HF_TOKEN] [WANDB_KEY]
# MODE: "full" (default) or "covid"
# Example: bash scripts/add_modal_embs.sh full 2 "hf_xxxxx" "wandb_xxxxx"
# Example: bash scripts/add_modal_embs.sh covid 2 "hf_xxxxx" "wandb_xxxxx"

MODE=${1:-full}
BATCH_SIZE=${2:-2}
HF_TOKEN=${3:-""}
WANDB_KEY=${4:-""}

# Set tasks and log directory based on mode
if [ "$MODE" = "covid" ]; then
    TRAIN_TASKS="S1,S2,S3,S4"
    TEST_TASKS="T1,T2,T3,T4"
    LOG_DIR="logs_llm_fixed_seed/COVID"
else
    TRAIN_TASKS="S1,S2,S3,S4,S5,S6,S7"
    TEST_TASKS="T1,T2,T3,T4,T5,T6"
    LOG_DIR="logs_llm_fixed_seed"
fi

mkdir -p "$LOG_DIR"
echo "Running in $MODE mode"
echo "  Train tasks: $TRAIN_TASKS"
echo "  Test tasks: $TEST_TASKS"
echo "  Log directory: $LOG_DIR"

encoder=("learnable" "llm_embeddings")

LLM_MODELS=("gemma2B")
NUM_RUNS=1

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

# get_batch_size() {
#     case "$1" in
#         GPT2) echo 16 ;;
#         gemma2B) echo 2 ;;
#         phi)     echo 1 ;;
#         mistral) echo 1 ;;
#         llama)   echo 1 ;;
#         deepseek-moe) echo 1 ;;
#         qwen-moe) echo 1 ;;
#     esac
# }

# ########################################
# # 3. FEATURE MODE 
# ########################################

# modal_dims=(32)
# feature_dims=(768)

# for enc in "${encoder[@]}"
# do
#     for i in ${!modal_dims[@]}
#     do
#     modal_dim=${modal_dims[$i]}
#     feature_dim=${feature_dims[$i]}

#     echo "Running FEATURE mode: modal=$modal_dim, feature=$feature_dim"

#     python src/benchmark/RespLLM/RespLLM.py \
#         $COMMON_ARGS \
#         --modality_encoder_type "$enc" \
#         --modal_embs projected_concat \
#         --out_modal_projector $modal_dim \
#         --out_feature_projector $feature_dim \
#         --wandb_name "projected_concat_m${modal_dim}_f${feature_dim}" \
#         2>&1 | tee -a logs_6/out_${LLM_MODEL}_${enc}_projected_concat_m${modal_dim}_f${feature_dim}.txt

#     done
# done 
########################################
# 2. RAW CONCAT MODE
########################################

echo "Running RAW CONCAT mode"
for MODEL in "${LLM_MODELS[@]}"; do
    LLM_DIM=$(get_llm_dim "$MODEL")
    # BATCH_SIZE=$(get_batch_size "$MODEL")
    for ((RUN=1;RUN<=NUM_RUNS; RUN++)); do    
        echo "Model: $MODEL | Run $RUN/$NUM_RUNS"
        for enc in "${encoder[@]}"; do
        LOG_FILE="$LOG_DIR/out_${MODEL}_${enc}_${RUN}_early_fusion.txt"

        if [ -f "$LOG_FILE" ]; then
            if grep -q "torch.cuda.OutOfMemoryError\|CUDA out of memory" "$LOG_FILE"; then
                echo "[OOM] Re-running $MODEL run $RUN (out of memory)"
                rm "$LOG_FILE"
            elif grep -q "Traceback\|Error" "$LOG_FILE"; then
                echo "[ERROR] Re-running $MODEL run $RUN (non-OOM error)"
                rm "$LOG_FILE"
            else
                echo "[SKIP] $MODEL run $RUN (assumed success)"
                continue
            fi
        fi
            echo "Running RAW CONCAT mode with encoder: $enc"
            run_name="raw_concat_${enc}_run${RUN}_${MODE}"
            python src/benchmark/RespLLM/RespLLM.py \
                --llm_model $MODEL \
                --train_tasks "$TRAIN_TASKS" \
                --test_tasks "$TEST_TASKS" \
                --train_epochs 60 \
                --meta_val_interval 3 \
                --meta_val_shot 20 \
                --train_pct 1 \
                --batch_size $BATCH_SIZE \
                --llm_dim ${LLM_DIM} \
                --d_ff ${LLM_DIM} \
                --modality_encoder_type "$enc" \
                --modal_embs raw_concat \
                --wandb_name "$run_name" \
                ${HF_TOKEN:+--hf_token "$HF_TOKEN"} \
                ${WANDB_KEY:+--wandb_key "$WANDB_KEY"} \
                2>&1 | tee -a "$LOG_FILE"
        done
    done
done

# for ((RUN=1;RUN<=NUM_RUNS; RUN++))
# do    echo "Run $RUN/$NUM_RUNS"
#     for enc in "${encoder[@]}"
#     do
#     LOG_FILE="logs_llm_fixed_valloss/out_${LLM_MODEL}_${enc}_${RUN}_add_val_S3_S4.txt"

#     if [ -f "$LOG_FILE" ]; then
#         echo "  [SKIP] Log file $LOG_FILE exists. Skipping $enc run $RUN."
#         continue
#     fi
#         echo "Running RAW CONCAT mode with encoder: $enc"
#         run_name="raw_concat_${enc}_run${RUN}_add_val_S3_S4"
#         python src/benchmark/RespLLM/RespLLM.py \
#             $COMMON_ARGS \
#             --modality_encoder_type "$enc" \
#             --modal_embs raw_concat \
#             --wandb_name "$run_name" \
#             --no-val_S3_S4 \
#             --save_pth cks/audio_type/"${LLM_MODEL}_${enc}_run${RUN}.pth" \
#             2>&1 | tee -a "$LOG_FILE"
#     done
# done

# NUM_RUNS=3
# for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
# for enc in "${encoder[@]}"; do
#     LOG_FILE="logs_24_4_26/out_AudioBert_${enc}_${RUN}.txt"

#     if [ -f "$LOG_FILE" ]; then
#         echo "  [SKIP] Log file $LOG_FILE exists. Skipping fusion run $RUN."
#         continue
#     fi

#     echo "  Fusion Model | Run: $RUN / $NUM_RUNS | batch: 32"
#     python src/benchmark/RespLLM/model_bts.py \
#         --train_tasks S1,S2,S3,S4,S5,S6,S7 \
#         --test_tasks T1,T2,T3,T4,T5,T6 \
#         --train_epochs 60 \
#         --batch_size 32 \
#         --modality_encoder_type "$enc" \
#         --modal_embs raw_concat \
#         --save_pth cks/fusion/fusion_run${RUN}_best_epoch.pth \
#         --wandb_name "fusion_run_${enc}" \
#         2>&1 | tee -a "$LOG_FILE"
# done
# done
