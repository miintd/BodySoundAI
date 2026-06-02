mkdir -p logs_llm_fix_seed/COVID
# LLM_MODELS=("gemma2B" "phi" "mistral" "llama" "deepseek-moe" "qwen-moe")
LLM_MODELS=("GPT2" "gemma2B")
NUM_RUNS=3
BATCH_SIZE=${1:-2}

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


for MODEL in "${LLM_MODELS[@]}"; do
    LLM_DIM=$(get_llm_dim "$MODEL")
    for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
        run_name="${MODEL}_run${RUN}"
        LOG_FILE="logs_llm_fix_seed/out_${MODEL}_${RUN}.txt"
        
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
            --train_tasks S1,S2,S3,S4,S5,S6,S7 \
            --test_tasks T1,T2,T3,T4,T5,T6 \
            --train_epochs 60 \
            --meta_val_interval 3 \
            --train_pct 1 \
            --batch_size "$BATCH_SIZE" \
            --llm_dim "$LLM_DIM" \
            --d_ff "$LLM_DIM" \
            --wandb_name "$run_name" \
            --use_8bit_quantization \
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