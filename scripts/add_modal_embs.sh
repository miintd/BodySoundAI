#!/bin/bash
mkdir -p logs_4

encoder=("llm_embeddings")

LLM_MODEL=${1:-gemma2B}
BATCH_SIZE=${2:-32}
LLM_DIM=${3:-2304}
D_FF=${4:-2304}
COMMON_ARGS="\
    --llm_model $LLM_MODEL \
    --train_tasks S1,S2,S3,S4,S5,S6,S7 \
    --test_tasks T1,T2,T3,T4,T5,T6 \
    --train_epochs 60 \
    --meta_val_interval 3 \
    --train_pct 1 \
    --batch_size $BATCH_SIZE \
    --llm_dim ${LLM_DIM} \
    --d_ff ${D_FF} \
"

########################################
# 1. NO MODALITY MODE
########################################

# echo "Running NO MODALITY mode"

# python src/benchmark/RespLLM/RespLLM.py \
#     $COMMON_ARGS \
#     --modal_embs none \
#     --wandb_name "no_modality" \
#     2>&1 | tee -a logs_4/out_${LLM_MODEL}_baseline.txt


# ########################################
# # 3. FEATURE MODE (grid search)
# ########################################

modal_dims=(64)
feature_dims=(768)

for enc in "${encoder[@]}"
do
    for i in ${!modal_dims[@]}
    do
    modal_dim=${modal_dims[$i]}
    feature_dim=${feature_dims[$i]}

    echo "Running FEATURE mode: modal=$modal_dim, feature=$feature_dim"

    python src/benchmark/RespLLM/RespLLM.py \
        $COMMON_ARGS \
        --modality_encoder_type "$enc" \
        --modal_embs projected_concat \
        --out_modal_projector $modal_dim \
        --out_feature_projector $feature_dim \
        --wandb_name "projected_concat_m${modal_dim}_f${feature_dim}" \
        2>&1 | tee -a logs_4/out_${LLM_MODEL}_${enc}_projected_concat_m${modal_dim}_f${feature_dim}.txt

    done
done 
########################################
# 2. RAW CONCAT MODE
########################################

# echo "Running RAW CONCAT mode"
# for enc in "${encoder[@]}"
# do
#     echo "Running RAW CONCAT mode with encoder: $enc"
#     run_name="raw_concat_${enc}"
#     python src/benchmark/RespLLM/RespLLM.py \
#         $COMMON_ARGS \
#         --modality_encoder_type "$enc" \
#         --modal_embs raw_concat \
#         --wandb_name "$run_name" \
#         2>&1 | tee -a logs_4/out_${LLM_MODEL}_${enc}_rawconcat_randw.txt
# done
