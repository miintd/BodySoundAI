mkdir -p logs_bts/COVID
TASKS=(S1 S2 S3 S4 S5 S6 S7)
encoder=("label")

# NUM_RUNS=3
# for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
#     LOG_FILE="logs_bts/COVID/out_AudioBert_${RUN}.txt"

#     if [ -f "$LOG_FILE" ]; then
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

#     echo "  Fusion Model | Run: $RUN / $NUM_RUNS"
#     python src/benchmark/RespLLM/model_bts.py \
#         --train_tasks "S1,S2,S3,S4" \
#         --test_tasks "T1,T2,T3,T4" \
#         --train_epochs 60 \
#         --batch_size 32 \
#         # --full_dataset True\
#         --wandb_name "fusion_run" \
#         2>&1 | tee -a "$LOG_FILE"

# done

# NUM_RUNS=1
# for TASK in "${TASKS[@]}"; do
#         LOG_FILE="logs_bts/fix_loss/out_AudioBert_${TASK}_${RUN}.txt"

#         if [ -f "$LOG_FILE" ]; then
#                 if grep -q "torch.cuda.OutOfMemoryError\|CUDA out of memory" "$LOG_FILE"; then
#                     echo "[OOM] Re-running $MODEL run $RUN"
#                     rm "$LOG_FILE"
#                 elif grep -q "Traceback\|Error" "$LOG_FILE"; then
#                     echo "[ERROR] Re-running $MODEL run $RUN (non-OOM error)"
#                     rm "$LOG_FILE"
#                 else
#                     echo "[SKIP] $MODEL run $RUN (assumed success)"
#                     continue
#                 fi
#             fi

#         echo "  Fusion Model | Run: $RUN / $NUM_RUNS | batch: 16"
#         python src/benchmark/RespLLM/model_bts.py \
#             --train_tasks $TASK \
#             --test_tasks $TASK \
#             --train_epochs 60 \
#             --batch_size 16 \
#             --wandb_name "fusion_run" \
#             2>&1 | tee -a "$LOG_FILE"

# done

# SOURCE_TASKS=("S2,S4" "S2,S4" "S1,S3" "S1,S3" "S7")
# TARGET_TASKS=("T1" "T2" "T3" "T4" "T5")
# for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
#     for i in "${!SOURCE_TASKS[@]}"; do
#         SRC="${SOURCE_TASKS[$i]}"
#         TGT="${TARGET_TASKS[$i]}"
#     LOG_FILE="logs_bts/fix_loss/out_AudioBert_${TGT}_${RUN}.txt"

#     if [ -f "$LOG_FILE" ]; then
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

#     echo "  Fusion Model | Run: $RUN / $NUM_RUNS | batch: 16"
#     python src/benchmark/RespLLM/model_bts.py \
#         --train_tasks $SRC \
#         --test_tasks $TGT \
#         --train_epochs 60 \
#         --batch_size 16 \
#         --wandb_name "fusion_run" \
#         2>&1 | tee -a "$LOG_FILE"
# done
# done

NUM_RUNS=1
for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
for enc in "${encoder[@]}"; do
    LOG_FILE="logs_bts/COVID/out_AudioBert_${enc}_${RUN}.txt"

    if [ -f "$LOG_FILE" ]; then
        echo "  [SKIP] Log file $LOG_FILE exists. Skipping fusion run $RUN."
        continue
    fi

    echo "  Fusion Model | Run: $RUN / $NUM_RUNS | batch: 32"
    python src/benchmark/RespLLM/model_bts.py \
        --train_tasks S1,S2,S3,S4 \
        --test_tasks T1,T2,T3,T4 \
        --train_epochs 60 \
        --batch_size 32 \
        --modality_encoder_type "$enc" \
        --modal_embs raw_concat \
        --wandb_name "fusion_run_${enc}" \
        2>&1 | tee -a "$LOG_FILE"
done
done

for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
for enc in "${encoder[@]}"; do
    LOG_FILE="logs_bts/COVID/out_AudioBert_${enc}_${RUN}_full.txt"

    if [ -f "$LOG_FILE" ]; then
        echo "  [SKIP] Log file $LOG_FILE exists. Skipping fusion run $RUN."
        continue
    fi

    echo "  Fusion Model | Run: $RUN / $NUM_RUNS | batch: 32"
    python src/benchmark/RespLLM/model_bts.py \
        --train_tasks S1,S2,S3,S4 \
        --test_tasks T1,T2,T3,T4 \
        --train_epochs 60 \
        --batch_size 32 \
        --modality_encoder_type "$enc" \
        --modal_embs raw_concat \
        --full_dataset True \
        --wandb_name "fusion_run_${enc}_full" \
        2>&1 | tee -a "$LOG_FILE"
done
done