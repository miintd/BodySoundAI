# mkdir -p logs_llm_fixed_valloss/eval

NUM_RUN=3
for ((RUN=1; RUN<=NUM_RUN; RUN++)); do
    LOG_FILE="logs_llm_fixed_valloss/eval/out_GPT2_${RUN}_ori_eval.txt"
    python src/benchmark/RespLLM/RespLLM_ori.py \
    --llm_model GPT2 \
    --train_tasks S1,S2,S3,S4,S5,S6,S7 \
    --test_tasks T1,T2,T3,T4,T5,T6 \
    --train_epochs 20 \
    --meta_val_interval 3 \
    --train_pct 1 \
    --batch_size 16 \
    --llm_dim 768 \
    --d_ff 768 \
    --from_pretrain True \
    --save_pth cks/llm_ori/model_GPT2_best_epoch_20260507-0915.pt \
    2>&1 | tee -a $LOG_FILE
done