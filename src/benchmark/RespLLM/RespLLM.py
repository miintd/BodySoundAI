import numpy as np
from transformers import AutoTokenizer
import transformers
import torch
from torch import nn
from torch.nn import functional as F
import os
import time
import collections
from tqdm import tqdm
import random
import requests
from sklearn.model_selection import train_test_split

from src.benchmark.RespLLM.model import RespLLM
from src.benchmark.RespLLM.util import test, test_and_collect_predictions, get_dataloader, EarlyStopper, set_all_seed, log_confusion_matrix_and_errors
# from src.benchmark.RespLLM.util import get_dataloader
from src.benchmark.RespLLM.sampler import CategoriesSampler

import wandb

# Optionally set WANDB_API_KEY programmatically
os.environ['WANDB_API_KEY'] = ""

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from pytorch_lightning.loggers import CSVLogger
# from lightning.pytorch import seed_everything
# from torchmetrics import AUROC
scaler = torch.cuda.amp.GradScaler()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def send_telegram_message(message):
    token = "8796712926:AAE3o-tQUbKUir61bRnCUGGii48kj_kGwWw"  
    chat_id = "8208495590"  
    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={message}"
    
    try:
        requests.get(url)
    except Exception as e:
        print(f"Lỗi gửi tin nhắn: {e}")


def train_RespLLM(configs):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loaders, val_loaders, test_loaders = [], [], []
    train_loaders_ = []
    n_cls = collections.defaultdict(lambda:2, {"11": 5, "12": 5})
    for task in configs.train_tasks:
        train_loader, val_loader, test_loader_ = get_dataloader(configs, task)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        train_loaders_.append(test_loader_)
    
    for task in configs.test_tasks:
        _, _, test_loader = get_dataloader(configs, task)
        test_loaders.append(test_loader)
    

    time_now = time.time() # tính thời gian training
    train_steps = len(train_loader) # tính số batch trong 1 epoch
    model = RespLLM(configs).to(DEVICE)
    #model.audio_encoder.to(DEVICE)
    #model.aligner.to(DEVICE)
    #model.output_projection.to(DEVICE)
    print(model.llm_model)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = torch.optim.Adam(trained_parameters, lr=configs.lr)
    loss_func = nn.CrossEntropyLoss()

    early_stopper = EarlyStopper(patience=5, min_delta=0.005)
    
    best_loss = float('inf')
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = f"cks/llm/{configs.llm_model}_{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write(str(configs))

    for epoch in tqdm(range(configs.train_epochs)):
        iter_count = 0
        train_loss = []

        iterators = [iter(dataloader) for dataloader in train_loaders]
        num_batch = [len(dataloader) for dataloader in train_loaders]

        model.train()
        epoch_time = time.time()

        i = 0
        # task_order = []
        # batch_count = {task: 0 for task in configs.train_tasks}
        while True:
            i += 1
            # Randomly select a dataloader
            # selected_idx = random.randint(0, len(configs.train_tasks) - 1)
            selected_idx = random.choices(range(len(configs.train_tasks)), weights=num_batch, k=1)[0]
            # print(f"Selected task index: {selected_idx}")
            selected_iterator = iterators[selected_idx]
            
            try:
                # Get the next batch from the selected dataloader
                x1, x2, x3, y, m = next(selected_iterator) # x1: audio, x2: metadata, x3: prompt, y: label, m: modality
            
            except StopIteration:
                # If any iterator is exhausted, break the loop
                break
            
            x1 = x1.to(DEVICE)
            y = y.to(DEVICE)
            
            iter_count += 1
            model_optim.zero_grad()

            y_hat = model(x1, x2, x3, m) 

            loss = loss_func(y_hat, y)
            train_loss.append(loss.item())

            if i < 3 or (i + 1) % 10 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((configs.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            
            loss.backward()
            model_optim.step()

            wandb.log({
                "train/loss": loss.item(),
                "epoch": epoch + 1,
                "iter": i
            })
        
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        print("train loss", train_loss)


        model.eval()

        # train
        # print("="*10 + "train set eval")
        # for j, train_loader in enumerate(train_loaders):
        #     test(model, train_loader, loss_func, configs.n_cls) #, plot_feature="llm/task" + configs.train_tasks[j] + "train")
        
        # validation
        print("="*10 + "validation")
        validation_loss = 0
        validation_aucs = []
        for j, val_loader in enumerate(val_loaders):
            print("Task", configs.train_tasks[j])
            val_acc, val_auc, val_loss = test(model, val_loader, loss_func, configs.n_cls, return_auc=True, print_cm=False) #, plot_feature="llm/task" + configs.train_tasks[j] + "val")
            validation_loss += val_loss
            validation_aucs.append(val_auc.item() if hasattr(val_auc, 'item') else val_auc)
            avg_val_loss = validation_loss / len(val_loaders)
        print("average validation loss", avg_val_loss)
        avg_val_auc = np.mean(validation_aucs) if validation_aucs else 0
        print("validation AUC", avg_val_auc)

        wandb.log({
            "val/loss": validation_loss / len(val_loaders),
            "val/auc": avg_val_auc,
            "epoch": epoch + 1
        })

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            configs.save_pth = os.path.join(run_dir, "model_best.pt")
            # configs.save_pth = f"cks/llm/model_{configs.llm_model}_best_epoch.pt"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_optim.state_dict(),
                # 'loss': validation_loss,
                }, configs.save_pth)
            print(f"New best model saved! (Loss: {best_loss:.7f})")

        if early_stopper.early_stop(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        # if (epoch + 1) % 10 == 0:
        #     checkpoint_path = f"cks/llm/model_{configs.llm_model}_epoch_{epoch+1}.pt"
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': model_optim.state_dict(),
        #     }, checkpoint_path)
        #     print(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")

            
        if (epoch + 1) % configs.test_interval == 0:

            print("="*10 + "test on seen tasks")
            for j, test_loader in enumerate(train_loaders_):
                print("Task", configs.train_tasks[j])
                
                print(f"Test loader {j} has {len(test_loader)} batches")
                if n_cls[configs.train_tasks[j]] != configs.n_cls:
                    pass
                else:
                    test(model, test_loader, loss_func, configs.n_cls)

            print("="*10 + "test on unseen tasks")
            for j, test_loader in enumerate(test_loaders):
                # test
                print("Task", configs.test_tasks[j])
                if  n_cls[configs.test_tasks[j]] != configs.n_cls:
                    # test(model, test_loader, loss_func, configs.n_cls, plot_feature="llm/task" + configs.test_tasks[j] + "test", plot_only=True)
                    pass
                else:
                    test(model, test_loader, loss_func, configs.n_cls) #, plot_feature="llm/task" + configs.test_tasks[j] + "test")

        # if (epoch + 1) % configs.meta_val_interval == 0:
        #     configs.save_pth = f"cks/llm/model_{configs.llm_model}_{epoch+1}epoch.pt"

        #     torch.save({
        #         'epoch': epoch,    
        #         'model_state_dict': model.state_dict(), 
        #         # 'loss': validation_loss,
        #         }, configs.save_pth)
            
        # # do_extract_and_evaluate(model, configs, train_loader=train_loader, que)
        # if early_stopper.early_stop(validation_loss):
        #     print("early stopping")      
        #     break
    
    # lưu mô hình đã train vào file checkpoint .pt
    # configs.save_pth = f"cks/llm/model_{configs.llm_model}_{epoch+1}epoch.pt"

    # torch.save({
    #     'epoch': epoch, # số epoch hiện tại
    #     'model_state_dict': model.state_dict(), # dict lưu tất cả trọng số mô hình
    #     # 'loss': validation_loss,
    #     }, configs.save_pth)
    wandb.finish()


def evaluate_RespLLM(configs):
    # Set seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    train_loaders, val_loaders, test_loaders, test_loaders_S = [], [], [], []
    n_cls = collections.defaultdict(lambda:2, {"11": 5, "12": 5})

    for task in configs.train_tasks:
        train_loader, val_loader, test_loader_S = get_dataloader(configs, task)
        test_loaders_S.append(test_loader_S)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    
    for task in configs.test_tasks:
        _, _, test_loader = get_dataloader(configs, task)
        test_loaders.append(test_loader)

    # tải model đã lưu lên và sử dụng để đánh giá
    checkpoint = torch.load(configs.save_pth)
    model = RespLLM(configs).to(DEVICE)
    # print(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    loss_func = nn.CrossEntropyLoss()

    # # train
    print("="*10 + "test on seen task")
    for j, test_loader_S in enumerate(test_loaders_S):
        task_name = configs.train_tasks[j]
        print("Task", task_name)
        if n_cls[task_name] != configs.n_cls:
            pass
        else:
            # # For S5/S6 (smoker detection), log detailed confusion matrix + misclassified samples
            # if task_name in ["S5", "S6"]:
            #     class_names = class_names_map.get(task_name, None)
            #     test_and_collect_predictions(model, test_loader_S, loss_func, configs.n_cls,
            #                                task_name=task_name, class_names=class_names,
            #                                output_dir="eval_logs", verbose=True)
            # else:
            #     test(model, test_loader_S, loss_func, configs.n_cls)
            test(model, test_loader_S, loss_func, configs.n_cls, print_cm=True)
    
    # # validation
    # print("="*10 + "validation")
    # validation_loss = 0
    # for j, val_loader in enumerate(val_loaders):
    #     validation_loss += test(model, val_loader, loss_func, configs.n_cls) #, plot_feature="llm/task" + configs.train_tasks[j] + "val")

    print("="*10 + "test on unseen task")
    for j, test_loader in enumerate(test_loaders):
        # test
        task_name = configs.test_tasks[j]
        print("Task", task_name)
        if n_cls[task_name] != configs.n_cls:
            pass
        else:
            test(model, test_loader, loss_func, configs.n_cls, print_cm=True) 



if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    parser.add_argument("--n_cls", type=int, default=2)

    parser.add_argument("--train_tasks", type=str, default="S1,S2,S3,S4,S5,S6,S7")
    parser.add_argument("--test_tasks", type=str, default="T1,T2,T3,T4,T5,T6")

    parser.add_argument("--audio_encoder", type=str, default="operaCT")
    parser.add_argument("--enc_dim", type=int, default=768)
    parser.add_argument("--audio_peft", type=str, default='frozen') # frozen, lora, IA3, full
    parser.add_argument("--audio_lora_rank",  type=int, default=8)

    parser.add_argument("--llm_model", type=str, default='llama')
    parser.add_argument("--use_8bit_quantization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--patch_nums', type=int, default=1, help='dimension of fcn') 
    parser.add_argument('--d_ff', type=int, default=2304, help='dimension of fcn') 
    #parser.add_argument('--patch_nums', type=int, default=64, help='dimension of fcn')
    #parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument("--llm_dim",  type=int, default=2304)
    parser.add_argument("--llm_peft", type=str, default='lora') # frozen, lora, IA3, full
    parser.add_argument("--llm_lora_rank",  type=int, default=16)
    parser.add_argument("--llm_lora_alpha",  type=int, default=32)
    parser.add_argument("--llm_lora_dropout",  type=float, default=0.1)
    parser.add_argument("--llm_lora_allproj",  type=bool, default=False)

    parser.add_argument("--aligner",  type=str, default="projection") # projection, Qformer, CNN, reprogram

    parser.add_argument("--head", type=str, default="linear")
    parser.add_argument("--head_dropout",  type=float, default=0)

    parser.add_argument("--n_run", type=int, default=5)
    parser.add_argument("--n_run_finetune", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_pct", type=float, default=1)
    parser.add_argument("--train_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--use_context", action=argparse.BooleanOptionalAction, default=True) #type=int, default=1)
    parser.add_argument("--use_context_", action=argparse.BooleanOptionalAction, default=True) 
    # parser.add_argument("--test_sampler", default=False)
    parser.add_argument("--use_audio", action=argparse.BooleanOptionalAction, default=True)
    
    parser.add_argument("--test_interval", type=int, default=1) #4
    parser.add_argument("--meta_val_interval", type=int, default=3) #4
    parser.add_argument("--meta_val_iter", type=int, default=10)
    parser.add_argument("--meta_val_way", type=int, default=5)
    parser.add_argument("--meta_val_shot", type=int, default=20)
    parser.add_argument("--meta_val_query", type=int, default=15)
    parser.add_argument("--meta_val_metric", type=str, default="euclidean") # euclidean, cosine, l1, l2

    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--use_L2N", type=bool, default=False)
    parser.add_argument("--use_centering", type=bool, default=False)

    parser.add_argument("--from_pretrain", type=bool, default=False)

    parser.add_argument("--save_pth", type=str, default="cks/llm/model_llama.pt")
    parser.add_argument('--eval_fc', action='store_true', help='do evaluate with final fc layer.')

    parser.add_argument("--few_shot", type=bool, default=False)
    parser.add_argument("--data_efficient_finetuning", type=bool, default=True)

    parser.add_argument("--wavelet", type=str, default=None) # None, db4, db8, sym4, sym8, coif2, coif3
    parser.add_argument("--wavelet_modality", type=str, default="cough") # cough, breath, lung, all
    parser.add_argument("--wavelet_level", type=int, default=4) # 4, 5, 6
    parser.add_argument("--threshold_mode", type=str, default="soft") # soft, hard
    parser.add_argument("--method", type=str, default="universal") # universal, bayesshrink, sureshrink
    parser.add_argument("--wandb_name", type=str, default="test")

    parser.add_argument("--use_audiolabel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--modal_embs", type=str, default=None)
    parser.add_argument("--out_modal_projector", type=int, default=1) 
    parser.add_argument("--out_feature_projector", type=int, default=10) 
    parser.add_argument("--modality_encoder_type", type=str, default="onehot") # cnn, transformer
    parser.add_argument("--modality_embed_dim", type=int, default=10) # chỉ dùng khi modality_encoder_type=learnable

    parser.add_argument("--val_S3_S4", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full_dataset", type=bool, default=False)
    
    # Credentials from arguments
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument("--wandb_key", type=str, default=None, help="Weights & Biases API key")

    args = parser.parse_args()
    print(args)
    
    # Set credentials from arguments
    if args.hf_token:
        os.environ['HF_TOKEN'] = args.hf_token
        from huggingface_hub import login
        login(token=args.hf_token, add_to_git_credential=False)
    
    if args.wandb_key:
        os.environ['WANDB_API_KEY'] = args.wandb_key

    args.train_tasks = args.train_tasks.split(",")
    args.test_tasks = args.test_tasks.split(",")

    if args.from_pretrain:
        evaluate_RespLLM(args)
    else:
        wandb.init(
            project=getattr(args, "wandb_project", "RespLLM_test_3"),
            name=getattr(args, "wandb_name", "test"),#"lowres_46_dynamic_"), #"Spread+MLP"
            config=vars(args)
        )
        train_RespLLM(args)

send_telegram_message("Mô hình RespLLM đã train xong.")