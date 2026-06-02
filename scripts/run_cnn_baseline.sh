#!/bin/bash

# CNN Baseline — Pure Audio Classification
# Uses mel spectrograms directly without LLM or audio encoder

NUM_RUNS=1
BATCH_SIZE=${1:-16}
TASKS=(S1 S2 S3 S4 S5 S6 S7)

mkdir -p logs_cnn_pretrained_1
mkdir -p cks/cnn

for TASK in "${TASKS[@]}"; do
for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
    run_name="cnn_${TASK}_run${RUN}"
    LOG_FILE="logs_cnn_pretrained_1/out_cnn_${TASK}_${RUN}.txt"

    if [ -f "$LOG_FILE" ]; then
        echo "  [SKIP] Log file $LOG_FILE exists. Skipping CNN run $RUN."
        continue
    fi

    echo "  CNN Baseline | Run: $RUN / $NUM_RUNS | batch: $BATCH_SIZE"

    python src/benchmark/RespLLM/cnn_train.py \
        --train_tasks $TASK \
        --test_tasks $TASK \
        --train_epochs 30 \
        --batch_size "$BATCH_SIZE" \
        --dropout 0.3 \
        --test_interval 1 \
        --save_pth "cks/cnn/cnn_${TASK}_run${RUN}_best_epoch.pth" \
        --wandb_name "$run_name" \
        2>&1 | tee -a "$LOG_FILE"

    echo "  Done: CNN run $RUN"
done
done

SOURCE_TASKS=("S2,S4" "S2,S4" "S1,S3" "S1,S3" "S7")
TARGET_TASKS=("T1" "T2" "T3" "T4" "T5")
for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
    for i in "${!SOURCE_TASKS[@]}"; do
        SRC="${SOURCE_TASKS[$i]}"
        TGT="${TARGET_TASKS[$i]}"
        
        run_name="cnn_zs_${TGT}_run${RUN}"
        LOG_FILE="logs_cnn_pretrained_1/out_cnn_${TGT}_run${RUN}.txt"

        if [ -f "$LOG_FILE" ]; then
            echo "  [SKIP] $TGT đã tồn tại. Bỏ qua."
            continue
        fi

        echo "  Huấn luyện trên [$SRC] -> Kiểm tra trên [$TGT]"

        # Chạy file training với các tham số nguồn và đích khác nhau
        python src/benchmark/RespLLM/cnn_train.py \
            --train_tasks $SRC \
            --test_tasks $TGT \
            --train_epochs 60 \
            --batch_size "$BATCH_SIZE" \
            --dropout 0.3 \
            --test_interval 1 \
            --save_pth "cks/cnn/cnn_${TGT}_run${RUN}_best_epoch.pth" \
            --wandb_name "$run_name" \
            2>&1 | tee -a "$LOG_FILE"

        echo "  Hoàn thành zero-shot cho $TGT"
    done
done


# NO PRETRAINED
# echo "Running CNN Baseline without pretrained weights"
# for TASK in "${TASKS[@]}"; do
# for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
#     run_name="cnn_no-pretrained_${TASK}_run${RUN}"
#     LOG_FILE="logs_cnn_pretrained_1/out_cnn_no-pretrained_${TASK}_${RUN}.txt"

#     if [ -f "$LOG_FILE" ]; then
#         echo "  [SKIP] Log file $LOG_FILE exists. Skipping CNN run $RUN."
#         continue
#     fi

#     echo "  CNN Baseline | Run: $RUN / $NUM_RUNS | batch: $BATCH_SIZE"

#     python src/benchmark/RespLLM/cnn_train.py \
#         --train_tasks $TASK \
#         --test_tasks $TASK \
#         --train_epochs 30 \
#         --batch_size "$BATCH_SIZE" \
#         --dropout 0.3 \
#         --test_interval 1 \
#         --no-pretrained \
#         --wandb_name "$run_name" \
#         2>&1 | tee -a "$LOG_FILE"

#     echo "  Done: CNN run $RUN"
# done
# done

# SOURCE_TASKS=("S2,S4" "S2,S4" "S1,S3" "S1,S3" "S7")
# TARGET_TASKS=("T1" "T2" "T3" "T4" "T5")
# for ((RUN=1; RUN<=NUM_RUNS; RUN++)); do
#     for i in "${!SOURCE_TASKS[@]}"; do
#         SRC="${SOURCE_TASKS[$i]}"
#         TGT="${TARGET_TASKS[$i]}"
        
#         run_name="cnn_zs_no-pretrained_${TGT}_run${RUN}"
#         LOG_FILE="logs_cnn/out_cnn_no-pretrained_${TGT}_run${RUN}.txt"

#         if [ -f "$LOG_FILE" ]; then
#             echo "  [SKIP] $TGT đã tồn tại. Bỏ qua."
#             continue
#         fi

#         echo "  Huấn luyện trên [$SRC] -> Kiểm tra trên [$TGT]"

#         # Chạy file training với các tham số nguồn và đích khác nhau
#         python src/benchmark/RespLLM/cnn_train.py \
#             --train_tasks $SRC \
#             --test_tasks $TGT \
#             --train_epochs 30 \
#             --batch_size "$BATCH_SIZE" \
#             --dropout 0.3 \
#             --test_interval 1 \
#             --no-pretrained \
#             --wandb_name "$run_name" \
#             2>&1 | tee -a "$LOG_FILE"

#         echo "  Hoàn thành zero-shot cho $TGT"
#     done
# done