#!/bin/bash

BATCH_SIZE=${1:-16}

mkdir -p logs_1

# no audio label
python src/benchmark/RespLLM/RespLLM.py \
        --llm_model gemma2B \
        --train_tasks S1,S2,S3,S4,S5,S6,S7 \
        --test_tasks T1,T2,T3,T4,T5,T6 \
        --train_epochs 40 \
        --meta_val_interval 3 \
        --train_pct 1 \
        --batch_size $BATCH_SIZE \
        2>&1 | tee -a logs_1/out_RespLLM_gemma2B.txt

# with audio label
python src/benchmark/RespLLM/RespLLM.py \
        --llm_model gemma2B \
        --train_tasks S1,S2,S3,S4,S5,S6,S7 \
        --test_tasks T1,T2,T3,T4,T5,T6 \
        --train_epochs 40 \
        --meta_val_interval 3 \
        --train_pct 1 \
        --batch_size $BATCH_SIZE \
        --use_audiolabel \
        2>&1 | tee -a logs_1/out_RespLLM_gemma2B_audiolabel.txt