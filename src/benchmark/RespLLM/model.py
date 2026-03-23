import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, BertModel, BertTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModel
import transformers
from peft import LoraConfig, TaskType, get_peft_model, IA3Config
import logging
from src.benchmark.model_util import get_encoder_path, initialize_pretrained_model

import pytorch_lightning as pl
from torchmetrics import AUROC

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()

token = "readacted"

# giúp xác định phần nào cần fine-tuning
OPERA_CT_TARGET_MODULES = ["qkv", "proj"]
OPERA_CE_TARGET_MODULES = ['conv', 'fc', 'linear']
target_module_dict = {"operaCT": OPERA_CT_TARGET_MODULES, "operaCE": OPERA_CE_TARGET_MODULES}
LLM_TARGET_MODULES = ["q_proj", "v_proj"]
LLM_TARGET_MODULES_ALLPROJ = ["q_proj", "k_proj", "v_proj", "o_proj"]

class FlattenHead(nn.Module):
    def __init__(self, nf, out_dim, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, out_dim)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x, no_fc=False):
        x = self.flatten(x)
        if no_fc:
            return x
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
class FeatureHead(nn.Module):
    def __init__(self, nf, out_feature_dim, head_dropout=0):
        super().__init__()
        self.proj = FlattenHead(nf, out_feature_dim, head_dropout)

    def forward(self, x, no_fc=False):
        return self.proj(x, no_fc=no_fc)
    
class ClassifierHead(nn.Module):
    def __init__(self, in_dim, n_cls):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_cls)

    def forward(self, x):
        return self.fc(x)

class RespLLM(nn.Module):
    
    def __init__(self, configs):
        super(RespLLM, self).__init__()

        self.loss = nn.CrossEntropyLoss() # hàm mất mát
        self.n_cls = configs.n_cls # số lớp phân loại
        self.validation_step_outputs = [] # lưu kết quả của bước validation sau mỗi 1 epoch
        self.test_step_outputs = [] # lưu kết quả của bước test sau mỗi 1 epoch

        self.d_ff = configs.d_ff # chiều đầu ra của lớp feed-forward
        self.d_llm = configs.llm_dim # chiều của embedding vector trong LLM
        # self.patch_len = configs.patch_len
        # self.stride = configs.
        self.audio_peft = configs.audio_peft # chế độ peft áp dụng cho audio
        self.d_audio = configs.enc_dim # chiều đầu ra của bộ mã hóa âm thanh
        self.patch_nums = configs.patch_nums # số lượng patch hoặc token LLM sử dụng cho việc phân loại cuối cùng
        self.head_nf = self.d_ff * self.patch_nums # chiều token * số token => kích thước đầu vào cho FlattenHead

        self.llm_peft = configs.llm_peft # fine-tuning cho LLM
        self.llm_lora_rank = configs.llm_lora_rank # kiểm soát kích thước các ma trận LoRA
        self.llm_lora_alpha = configs.llm_lora_alpha # điều chỉnh tỉ lệ học của các ma trận LoRA
        self.llm_lora_dropout = configs.llm_lora_dropout # tỉ lệ dropout cho các ma trận LoRA

        self.use_audio = configs.use_audio # boolean xác định mô hình có sử dụng đầu vào audio trong FF không
        self.modal_embs = configs.modal_embs # boolean xác định có sử dụng embedding của modality hay không

        if configs.llm_model == 'llama':
            # self.llama_config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B') 
            # hàm tải cấu hình của mô hình pre-trained huggyllama
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b') # 13.5G
            
            try: # tải mô hình llama từ trong máy
                self.llm_model = LlamaModel.from_pretrained(
                    # "meta-llama/Meta-Llama-3-8B",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                # nếu không tải được từ máy thì tải từ Hugging Face
                self.llm_model = LlamaModel.from_pretrained(
                    # "meta-llama/Meta-Llama-3-8B",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try: # tải tokenizer (biến văn bản thành chuỗi token)
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "meta-llama/Meta-Llama-3-8B",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "meta-llama/Meta-Llama-3-8B",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'llama2':
            # self.llama_config = LlamaConfig.from_pretrained('meta-llama/Meta-Llama-3-8B')
            # self.llama_config = LlamaConfig.from_pretrained('meta-llama/Llama-2-7b') # 13.5G
            model_id = "meta-llama/Llama-2-7b-chat-hf"
            # model_id = 'meta-llama/Llama-2-7b'
            self.llama_config = LlamaConfig.from_pretrained(model_id, token=token)
            self.tokenizer = LlamaTokenizer.from_pretrained(model_id, token=token)
            self.llm_model = LlamaModel.from_pretrained(model_id, token=token, config=self.llama_config)
        elif configs.llm_model == 'medalpaca':
            self.llama_config = LlamaConfig.from_pretrained("medalpaca/medalpaca-7b") # 13.5G
            
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    "medalpaca/medalpaca-7b",
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    "medalpaca/medalpaca-7b",
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    "medalpaca/medalpaca-7b",
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    "medalpaca/medalpaca-7b",
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == "OpenBioLLM":
            model_id = "aaditya/OpenBioLLM-Llama3-8B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = AutoModel.from_pretrained(model_id)
            # self.llama_config = LlamaConfig.from_pretrained(model_id) # 13.5G
            
            # try:
            #     self.llm_model = LlamaModel.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=True,
            #         config=self.llama_config,
            #         # load_in_4bit=True
            #     )
            # except EnvironmentError:  # downloads model from HF is not already done
            #     print("Local model files not found. Attempting to download...")
            #     self.llm_model = LlamaModel.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=False,
            #         config=self.llama_config,
            #         # load_in_4bit=True
            #     )
            # try:
            #     self.tokenizer = LlamaTokenizer.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=True
            #     )
            # except EnvironmentError:  # downloads the tokenizer from HF if not already done
            #     print("Local tokenizer files not found. Atempting to download them..")
            #     self.tokenizer = LlamaTokenizer.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=False
            #     )
        elif configs.llm_model == "llama3":
            model_id = "meta-llama/Meta-Llama-3-8B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.llm_model = AutoModel.from_pretrained(model_id)
            # self.llama_config = LlamaConfig.from_pretrained(model_id) # 13.5G
            
            # try:
            #     self.llm_model = LlamaModel.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=False,
            #         config=self.llama_config,
            #         # load_in_4bit=True
            #     )
            # except EnvironmentError:  # downloads model from HF is not already done
            #     print("Local model files not found. Attempting to download...")
            #     self.llm_model = LlamaModel.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=False,
            #         config=self.llama_config,
            #         # load_in_4bit=True
            #     )
            # try:
            #     self.tokenizer = LlamaTokenizer.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=False
            #     )
            # except EnvironmentError:  # downloads the tokenizer from HF if not already done
            #     print("Local tokenizer files not found. Atempting to download them..")
            #     self.tokenizer = LlamaTokenizer.from_pretrained(
            #         model_id,
            #         trust_remote_code=True,
            #         local_files_only=False
            #     )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            # self.gpt2_config.num_hidden_layers = configs.llm_layers
            # self.gpt2_config.output_attentions = True
            # self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            # self.bert_config.num_hidden_layers = configs.llm_layers
            # self.bert_config.output_attentions = True
            # self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        # elif configs.llm_model == 'mistral':
        #     model_id = "/home/tran-dam-quoc-khanh/Documents/DIEM_MY/Mistral-7B-v0.1"
        #     self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        #     self.llm_model = AutoModel.from_pretrained(model_id, local_files_only=True, load_in_4bit=True)
        # elif configs.llm_model == 'phi':
        #     model_id = "/home/tran-dam-quoc-khanh/Documents/DIEM_MY/Phi-3.5-mini-instruct"
        #     self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        #     self.llm_model = AutoModel.from_pretrained(model_id, local_files_only=True, load_in_8bit=True)
        # elif configs.llm_model == "gemma2B":
        #     # model_id = "google/gemma-2-2b-it"
        #     model_id = "/home/tran-dam-quoc-khanh/Documents/DIEM_MY/gemma-2-2b"
        #     self.tokenizer = AutoTokenizer.from_pretrained(model_id,local_files_only=True)
        #     #self.llm_model = AutoModel.from_pretrained(model_id)
        #     self.llm_model = AutoModel.from_pretrained(model_id, local_files_only=True, device_map="auto", load_in_8bit=True)
        elif configs.llm_model == 'mistral':
            model_id = "mistralai/Mistral-7B-v0.1"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token = token)
            self.llm_model = AutoModel.from_pretrained(model_id, token = token)
        elif configs.llm_model == 'phi':
            model_id = "microsoft/Phi-3.5-mini-instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token = token)
            self.llm_model = AutoModel.from_pretrained(model_id, token = token)
        elif configs.llm_model == "gemma2B":
            # model_id = "google/gemma-2-2b-it"
            model_id = "google/gemma-2-2b"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token = token)
            self.llm_model = AutoModel.from_pretrained(model_id, token = token)
        elif configs.llm_model == "gemma9B":
            # model_id = "google/gemma-2-9b-it"
            model_id = "google/gemma-2-9b"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token = token)
            self.llm_model = AutoModel.from_pretrained(model_id, token = token)
        else:
            raise NotImplementedError('LLM model is not defined')
        # thêm pad token để các token đưa vào có cùng độ dài
        if self.tokenizer.eos_token: # token để kết thúc câu
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # áp dụng LoRA cho các model
        if self.llm_peft == "lora":
            # khởi tạo cấu hình LoRA
            self.peft_config = LoraConfig(
                r=self.llm_lora_rank, 
                lora_alpha=self.llm_lora_alpha, 
                lora_dropout=self.llm_lora_dropout,
                # target_modules=LLM_TARGET_MODULES_ALLPROJ,
            )
            if configs.llm_lora_allproj:
                self.peft_config = LoraConfig(
                    r=self.llm_lora_rank, 
                    lora_alpha=self.llm_lora_alpha, 
                    lora_dropout=self.llm_lora_dropout,
                    target_modules=LLM_TARGET_MODULES_ALLPROJ,
                )
            try:
                self.llm_model = get_peft_model(self.llm_model, self.peft_config)
            except ValueError:
                print(self.llm_model)
                if configs.llm_model == "phi":
                    self.peft_config = LoraConfig(
                        r=self.llm_lora_rank, 
                        lora_alpha=self.llm_lora_alpha, 
                        lora_dropout=self.llm_lora_dropout,
                        target_modules=["qkv_proj"]
                    )
                else:
                    self.peft_config = LoraConfig(
                        r=self.llm_lora_rank, 
                        lora_alpha=self.llm_lora_alpha, 
                        lora_dropout=self.llm_lora_dropout,
                        target_modules=LLM_TARGET_MODULES
                    )
                self.llm_model = get_peft_model(self.llm_model, self.peft_config)
            self.llm_model.print_trainable_parameters()
            print('LoRA Training LLM')
        # áp dụng frozen cho các tham số được chọn
        elif self.llm_peft == "frozen":
            for param in self.llm_model.parameters():
                param.requires_grad = False
        else:
            return NotImplementedError("LLM fine-tuning mode undefined")
        
        if configs.audio_encoder == "operaCT": # tải mô hình operaCT và chỉ lấy phần encoder
            self.audio_encoder = initialize_pretrained_model(configs.audio_encoder).encoder

        # fine-tuning audio
        if self.audio_peft == "frozen":
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False # tất cả parameter đều ko được tính gradient, tức là ko được train
            self.audio_encoder.eval()
            print("freeze audio encoder")
        elif self.audio_peft == "full":
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = True # tất cả parameter trong audio encoder đều được cập nhật, fine-tune hoàn toàn
            self.audio_encoder.train()
            print("full model fine-tune audio encoder")
        else:
            # peft
            if self.audio_peft == "lora":
                peft_config = LoraConfig(
                    # task_type=TaskType.CAUSAL_LM, inference_mode=False, 
                    r=configs.audio_lora_rank, lora_alpha=32, lora_dropout=0.1,
                    target_modules=target_module_dict[configs.audio_encoder]
                )
            elif self.audio_peft == "IA3":
                peft_config = IA3Config(
                    target_modules=target_module_dict[configs.audio_encoder],
                    feedforward_modules=['proj']
                )
            else:
                return NotImplementedError("audio fine-tuning mode undefined")
            self.audio_encoder = get_peft_model(self.audio_encoder, peft_config)
            self.audio_encoder.print_trainable_parameters()
            
        # áp dụng projector để căn chỉnh chiều âm thanh giống với chiều của llm
        if configs.aligner == "projection":
            self.aligner = nn.Linear(self.d_audio, self.d_llm)
        else:
            return NotImplementedError("aligner module undefined")
        
        # modality_classes = ["exhalation", "cough", "breath", "lung", "cough-shallow", "cough-heavy", "breathing-shallow", "breathing-deep"]
        modality_classes = ["school", "home", "dog", "outdoor", "transportation", "office", "gym", "cat"]
        self.modality_encoder_type = configs.modality_encoder_type  # "onehot" | "bert"
        self.modality_classes = modality_classes
        self.num_modalities = len(modality_classes)         # 8
        self.modality2idx = {m: i for i, m in enumerate(modality_classes)}
        
        self.head_dropout = configs.head_dropout
        self.out_modal_projector = configs.out_modal_projector
        self.out_feature_projector = configs.out_feature_projector
        self.feature_head = FeatureHead(self.head_nf, self.out_feature_projector, head_dropout=self.head_dropout)
        self.modality_projector = nn.Linear(self.llm_model.config.hidden_size, self.out_modal_projector)
        self.classifier = ClassifierHead(self.out_feature_projector+self.out_modal_projector, self.n_cls)
        if self.modality_encoder_type == "onehot":
            modality_dim = self.num_modalities          # thường = 8
        elif self.modality_encoder_type == "bert":
            self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = BertModel.from_pretrained("bert-base-uncased")
            for param in self.bert_model.parameters():
                param.requires_grad = False
            modality_dim = 768                          # BERT hidden size
        elif self.modality_encoder_type == "llm_embeddings":
            modality_dim = self.llm_model.config.hidden_size
        self.classifier_raw = ClassifierHead(modality_dim + self.llm_model.config.hidden_size, self.n_cls)

        self.output_projection = FlattenHead(self.head_nf, self.n_cls, head_dropout=self.head_dropout)
        self.print_trainable()

    def reinitialize_clf(self, n_cls): # định nghĩa lại lớp đầu ra 
        self.output_projection = FlattenHead(self.head_nf, n_cls, head_dropout=self.head_dropout)

    def print_trainable(self): # in tổng số lượng tham số được train
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("total trainable parameters:", trainable_params)

    def reset_trainable(self): # đặt lại trạng thái cho các tham số mỗi lần train model
        if self.llm_peft == "lora":
            for name, param in self.audio_encoder.named_parameters(): ### nên là self.llm_model
                if "lora" in name: #chỉ train lora parameter
                    param.requires_grad = True 
                else: # ko train các tham số còn lại
                    param.requires_grad = False
        elif self.llm_peft == "frozen":
            for param in self.llm_model.parameters():
                param.requires_grad = False # ko train toàn bộ tham số của LLM
        
        for param in self.aligner.parameters():
            param.requires_grad = True # train tham số của projector
        
        if self.audio_peft == "frozen": 
            for param in self.audio_encoder.parameters():
                param.requires_grad = False # ko train tham số của audio encoder
        elif self.audio_peft == "full":
            for param in self.audio_encoder.parameters():
                param.requires_grad = True
        
        for param in self.output_projection.parameters():
            param.requires_grad = True # train tham số của lớp Linear
        self.print_trainable()

    def forward1(self, x_spectrogram, x_prompt, x_context, no_fc=False):
        target_device = self.llm_model.get_input_embeddings().weight.device 
        target_dtype = self.llm_model.get_input_embeddings().weight.dtype
        device = next(self.audio_encoder.parameters()).device
        x_spectrogram = x_spectrogram.to(device)
    
        if self.patch_nums == 1:
            x_enc = self.audio_encoder(x_spectrogram)
            # print(x_enc.shape)
            enc_out = self.aligner(x_enc)
            enc_out = enc_out.unsqueeze(dim=1)
        elif self.patch_nums == 64:
            x_enc = self.audio_encoder.forward_window(x_spectrogram)
            # print(x_enc.shape)
            enc_out = self.aligner(x_enc.to(target_dtype))
        else:
            raise NotImplementedError

        prompt = self.tokenizer(x_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids.to(target_device)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt)  # (batch, prompt_token, dim)

        context = self.tokenizer(x_context, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids.to(target_device)
        context_embeddings = self.llm_model.get_input_embeddings()(context)  # (batch, prompt_token, dim)

        enc_out = enc_out.to(device=target_device)  
        
        if self.use_audio:
            llama_enc_out = torch.cat([prompt_embeddings, context_embeddings, enc_out], dim=1)
        else:
            llama_enc_out = torch.cat([prompt_embeddings, context_embeddings], dim=1)

        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        # print(dec_out.shape)
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        # print(dec_out.shape)
        dec_out = self.output_projection(dec_out[:, :, -self.patch_nums:], no_fc=no_fc)
        return dec_out
    

    def _encode_modality(self, x_modality, device):
        if self.modality_encoder_type == "onehot":
            import torch.nn.functional as F

            indices_tensor = torch.tensor(
                [self.modality2idx.get(m, 0) for m in x_modality],
                device=device
            )
            modality_vec = F.one_hot(indices_tensor, num_classes=self.num_modalities).float() # (batch, 8)
            
        elif self.modality_encoder_type == "bert":
            # Encode từng modality string qua BERT, lấy [CLS] token
            tokens = self.bert_tokenizer(x_modality,return_tensors="pt",padding=True,truncation=True,max_length=16).to(device)
            with torch.no_grad():
                bert_out = self.bert_model(**tokens)
            modality_vec = bert_out.last_hidden_state[:, 0, :]   # (batch, 768)     
        
        elif self.modality_encoder_type == "llm_embeddings":
            modality = self.tokenizer(x_modality, return_tensors="pt", padding=True, truncation=True, max_length=16).input_ids.to(device)
            modality_embeddings = self.llm_model.get_input_embeddings()(modality)  # (batch, n_token, hidden_size)
            modality_vec = modality_embeddings.mean(dim=1)           # (batch, hidden_size)

        return modality_vec 

    def forward(self, x_spectrogram, x_prompt, x_context, x_modality, no_fc=False):
        if self.patch_nums == 1:
            x_enc = self.audio_encoder(x_spectrogram)
            # print("audio shape before aligner:", x_enc.shape)
            enc_out = self.aligner(x_enc)
            enc_out = enc_out.unsqueeze(dim=1)
        elif self.patch_nums == 64:
            x_enc = self.audio_encoder.forward_window(x_spectrogram)
            # print(x_enc.shape)
            enc_out = self.aligner(x_enc)
        else:
            raise NotImplementedError

        prompt = self.tokenizer(x_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        context = self.tokenizer(x_context, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        context_embeddings = self.llm_model.get_input_embeddings()(context.to(x_enc.device))  # (batch, prompt_token, dim)

        if self.modal_embs is not None:
            modality_vec = self._encode_modality(x_modality, x_enc.device)   # (batch, hidden_size)

        # print("audio shape after aligner:", enc_out.shape)
        # print("prompt_embeddings shape:", prompt_embeddings.shape)
        # print("context_embeddings shape:", context_embeddings.shape)
        # print("modality_vector shape:", modality_vec.shape)

        if self.use_audio:
            llama_enc_out = torch.cat([prompt_embeddings, context_embeddings, enc_out], dim=1)
        else:
            llama_enc_out = torch.cat([prompt_embeddings, context_embeddings], dim=1)

        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        # print("dec_out shape:", dec_out.shape)
        dec_out = dec_out[:, :, :self.d_ff]
        # print("dec_out shape after slicing:", dec_out.shape)
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        # print("dec_out shape before flatten head:", dec_out.shape)

        if self.modal_embs == "projected_concat":
            # print("Using modality embeddings for classification")
            dec_out = self.output_projection(dec_out[:, :, -self.patch_nums:], no_fc=True)
            # print("dec_out shape after flatten head:", dec_out.shape)
            modality_vec = self.modality_projector(modality_vec)      # (batch, 10)
            # print("modality_vec shape:", modality_vec.shape)
            fused = torch.cat([dec_out, modality_vec], dim=1)      
            # print("dec_out shape after adding modality vector:", dec_out.shape)
            dec_out = self.classifier(fused)                          # (batch, 2)
            # print("dec_out shape after final fc:", dec_out.shape)
        elif self.modal_embs == "raw_concat":
            dec_flat = self.output_projection(dec_out[:, :, -self.patch_nums:], no_fc=True)
            # print("dec_out shape after flatten head:", dec_flat.shape)
            # print("modality_vec shape:", modality_vec.shape)
            fused = torch.cat([dec_flat, modality_vec], dim=1)
            # print("dec_out shape after adding modality vector:", fused.shape)
            dec_out = self.classifier_raw(fused)
            # print("dec_out shape after final fc:", dec_out.shape)
        elif self.modal_embs == None:
            # print("Not using modality embeddings for classification")
            dec_out = self.output_projection(dec_out[:, :, -self.patch_nums:], no_fc=no_fc)

        return dec_out
    