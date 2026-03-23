# RespLLM


This is the code used for the paper: RespLLM: Unifying Audio and Text with Multimodal LLMs for Generalized Respiratory Health Prediction.

This code utilize the [OPERA framework](https://github.com/evelyn0414/OPERA)[1] for the audio dataset downloading and processing, and you can put this directory under `src/benchmark`. 

example for training RespLLM:
```
python src/benchmark/RespLLM/RespLLM.py --llm_model GPT2 --train_tasks S1,S2 --test_tasks S5,S6 --train_epochs 10 --meta_val_interval 3  --train_pct 1 --batch_size 16 --llm_dim 768 --d_ff 768 >> out_RespLLM_GPT2.txt
```


[1] Zhang, Yuwei, et al. "Towards Open Respiratory Acoustic Foundation Models: Pretraining and Benchmarking." arXiv preprint arXiv:2406.16148 (2024).
