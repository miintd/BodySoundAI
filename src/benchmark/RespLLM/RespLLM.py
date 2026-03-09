# file quản lý quy trình học máy: tải dữ liệu, khởi tạo mô hình/bộ tối ưu hóa, vòng lặp huấn luyện, và đánh giá hiệu suất
import requests
from pyexpat import model
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
from sklearn.model_selection import train_test_split

from src.benchmark.RespLLM.model import RespLLM
from src.benchmark.RespLLM.util_ori import test, get_dataloader, EarlyStopper, set_all_seed
# from src.benchmark.RespLLM.util import get_dataloader
from src.benchmark.RespLLM.sampler import CategoriesSampler


# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from pytorch_lightning.loggers import CSVLogger
# from lightning.pytorch import seed_everything
# from torchmetrics import AUROC
scaler = torch.cuda.amp.GradScaler()
token = "redacted"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_RespLLM(configs):
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

    early_stopper = EarlyStopper(patience=2, min_delta=0.01)
    
    best_loss = float('inf')

    for epoch in tqdm(range(configs.train_epochs)):
        iter_count = 0
        train_loss = []

        iterators = [iter(dataloader) for dataloader in train_loaders]
        num_batch = [len(dataloader) for dataloader in train_loaders]

        model.train()
        epoch_time = time.time()

        i = 0
        while True:
            i += 1
            # Randomly select a dataloader
            # selected_idx = random.randint(0, len(configs.train_tasks) - 1)
            selected_idx = random.choices(range(len(configs.train_tasks)), weights=num_batch, k=1)[0]
            selected_iterator = iterators[selected_idx]
            
            try:
                # Get the next batch from the selected dataloader
                x1, x2, x3, y = next(selected_iterator)
            except StopIteration:
                # If any iterator is exhausted, break the loop
                break
            
            #print(x1.dtype)
            #audio_device = next(model.audio_encoder.parameters()).device
            #x1 = move_to_device(x1, audio_device)
            x1 = x1.to(DEVICE)
            y = y.to(DEVICE)
            #llm_device = model.llm_model.get_input_embeddings().weight.device
            #x2 = move_to_device(x2, llm_device)
            #x3 = move_to_device(x3, llm_device)
            iter_count += 1
            model_optim.zero_grad()

            y_hat = model(x1, x2, x3) 
            #y = move_to_device(y, y_hat.device).long()
            loss = loss_func(y_hat, y)
            train_loss.append(loss.item())
            
            #model_optim.zero_grad(set_to_none=True)

            #with torch.cuda.amp.autocast(dtype=torch.float16):
            #    y_hat = model(x1, x2, x3)
            #    loss = loss_func(y_hat, y)
            
            #train_loss.append(loss.detach().item())
            #scaler.scale(loss).backward()
            #scaler.step(model_optim)
            #scaler.update()
            #print(y_hat.dtype)
            #print(loss.dtype)


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
        for j, val_loader in enumerate(val_loaders):
            validation_loss += test(model, val_loader, loss_func, configs.n_cls) #, plot_feature="llm/task" + configs.train_tasks[j] + "val")
        print("cumulative validation loss", validation_loss)

        wandb.log({
            "val/loss": validation_loss / len(val_loaders),
            "epoch": epoch + 1
        })

        if validation_loss < best_loss:
            best_loss = validation_loss
            
            configs.save_pth = f"cks/llm/model_{configs.llm_model}_best_epoch.pt"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_optim.state_dict(),
                # 'loss': validation_loss,
                }, configs.save_pth)
            print(f"New best model saved! (Loss: {best_loss:.7f})")

        if early_stopper.early_stop(validation_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        if (epoch + 1) % configs.test_interval == 0:

            print("="*10 + "test on seen tasks")
            for j, test_loader in enumerate(train_loaders_):
                print("Task", configs.train_tasks[j])
                
                print(f"Test loader {j} has {len(test_loader)} batches")
                if n_cls[configs.train_tasks[j]] != configs.n_cls:
                    pass
                else:
                    test(model, test_loader, loss_func, configs.n_cls)

            print("="*10 + "test")
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

def train_RespLLM_mixup(configs):
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
        
    

    time_now = time.time()
    train_steps = len(train_loader)
    
    if configs.save_pth_pretrain is not None:
        print("Loading pretrained embedding model from", configs.save_pth_pretrain)
        checkpoint = torch.load(configs.save_pth_pretrain)
        # model = RespLLM(configs)
        if configs.model_type == "original":
            model = RespLLM_original(configs).to(DEVICE)
            print("Using original RespLLM model for evaluation.")
        else:
            model = RespLLM(configs).to(DEVICE)
        
        # # model = RespLLM(configs).to(DEVICE)
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs for DataParallel")
        #     model = nn.DataParallel(model)
        # model = model.to(DEVICE)

        state_dict = checkpoint['model_state_dict']
        # remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[len("module."):]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        
        # now wrap with DataParallel if needed
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for DataParallel")
            model = nn.DataParallel(model)

        model = model.to(DEVICE)
        print("Pretrained embedding model loaded.")
        
    else:
        print("Training from scratch.")
    
        # model = RespLLM(configs).to(DEVICE)
        # model = RespLLM(configs)
        if configs.model_type == "original":
            model = RespLLM_original(configs).to(DEVICE)
            print("Using original RespLLM model for evaluation.")
        else:
            model = RespLLM(configs).to(DEVICE)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for DataParallel")
            model = nn.DataParallel(model)
        model = model.to(DEVICE)

    
    # print(model.llm_model)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    # model_optim = torch.optim.Adam(trained_parameters, lr=configs.lr)
    model_optim = torch.optim.Adam(trained_parameters, lr=configs.lr, weight_decay=0.0001)
    # KL Divergence Loss is required for mixed/soft labels [cite: 295]
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    # Keep standard loss for validation/testing
    criterion_ce = nn.CrossEntropyLoss()
    
    loss_func = nn.CrossEntropyLoss()

    early_stopper = EarlyStopper(patience=5, min_delta=0.01)
    
    
    
    # start = checkpoint['epoch'] + 1  # if continue training from checkpoint
    # model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = float('inf')
    # for epoch in tqdm(range(start, configs.train_epochs)):
    for epoch in tqdm(range(configs.train_epochs)):
        
        # if (epoch + 1) == 2:
        #     #stop running after 1 epochs for testing
        #     print("Stopping with tweak =", configs.tweak)
        #     break
        
        iter_count = 0
        train_loss = []

        iterators = [iter(dataloader) for dataloader in train_loaders]
        num_batch = [len(dataloader) for dataloader in train_loaders]

        model.train()
        epoch_time = time.time()

        i = 0
        while True:
            i += 1
            # Randomly select a dataloader
            # selected_idx = random.randint(0, len(configs.train_tasks) - 1)
            selected_idx = random.choices(range(len(configs.train_tasks)), weights=num_batch, k=1)[0]
            selected_iterator = iterators[selected_idx]
            
            try:
                # Get the next batch from the selected dataloader
                x1, x2, x3, y = next(selected_iterator)
                # x1, x2, x3, y = next(iterators[1         ])
            except StopIteration:
                # If any iterator is exhausted, break the loop
                break
            
            
            def get_shape(x):
                if x is None:
                    return None
                elif torch.is_tensor(x):
                    return tuple(x.shape)
                elif isinstance(x, (list, tuple)):
                    try:
                        # check if it’s a list of tensors
                        return [tuple(t.shape) for t in x]
                    except Exception:
                        return f"{type(x)} len={len(x)}"
                else:
                    return type(x)

            # print(f"  x1: {get_shape(x1)}")
            # print(f"  x2: {get_shape(x2)}")
            # print(f"  x3: {get_shape(x3)}")
            # print(f"  y: {get_shape(y)}")
            # print("="*60)
            # asdasd
            
            
            x1 = x1.to(DEVICE)
            y = y.to(DEVICE)
            
            
            # --- START MIXUP AUGMENTATION [cite: 281-294] ---
            # 1. Convert integer labels to one-hot for mixing
            y_one_hot = torch.nn.functional.one_hot(y, num_classes=configs.n_cls).float()
            
            # 2. Apply Mixup to double the data size [cite: 293]
            # This generates X_mp1, X_mp2 and y_mp1, y_mp2 as per Eqs 10-13
            mixed_x1, mixed_y = apply_paper_mixup(x1, y_one_hot)
                                                #   , task=configs.task_type)
            
            # Note: If x2 and x3 are metadata/prompts, they are replicated 
            # to match the doubled batch size
            
            # mixed_x2 = x2 * 2 if isinstance(x2, list) else torch.cat([x2, x2], dim=0)
            # mixed_x3 = x3 * 2 if isinstance(x3, list) else torch.cat([x3, x3], dim=0)
            
            mixed_x2 = x2 + x2 if isinstance(x2, (list, tuple)) else torch.cat([x2, x2], dim=0)
            mixed_x3 = x3 + x3 if isinstance(x3, (list, tuple)) else torch.cat([x3, x3], dim=0)

            # --- END MIXUP AUGMENTATION ---
            
            
            iter_count += 1
            model_optim.zero_grad()

            # y_hat = model(x1, x2, x3)
            # Forward pass with augmented data
            y_hat = model(mixed_x1, mixed_x2, mixed_x3)
            
            # loss = loss_func(y_hat, y)
            
            # Use KL-Divergence Loss for mixed labels 
            # KLDiv expects log-probabilities for the input
            loss = criterion_kl(torch.log_softmax(y_hat, dim=1), mixed_y)
            
            train_loss.append(loss.item())

            if i < 3 or (i + 1) % 250 == 0:
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

        # # train
        # print("="*10 + "train set eval")
        # for j, train_loader in enumerate(train_loaders):
        #     test(model, train_loader, loss_func, configs.n_cls) #, plot_feature="llm/task" + configs.train_tasks[j] + "train")
        
        # validation
        print("="*10 + "validation")
        validation_loss = 0
        for j, val_loader in enumerate(val_loaders):
            validation_loss += test(model, val_loader, loss_func, configs.n_cls) #, plot_feature="llm/task" + configs.train_tasks[j] + "val")
        
        print("cumulative validation loss", validation_loss)
        
        wandb.log({
            "val/loss": validation_loss / len(val_loaders),
            "epoch": epoch + 1
        })
        
        # --- SAVE LOGIC: Only save if this is the best model so far ---
        if validation_loss < best_loss:
            best_loss = validation_loss
            
            configs.save_pth = f"cks/original/model_{configs.llm_model}_best_epoch.pt"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_optim.state_dict(),
                # 'loss': validation_loss,
                }, configs.save_pth)
            print(f"New best model saved! (Loss: {best_loss:.7f})")

        # --- STOP LOGIC: Check if we need to stop ---
        if early_stopper.early_stop(validation_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        if (epoch + 1) % configs.test_interval == 0:
            
            
            
            print("="*10 + "test on seen tasks")
            for j, test_loader in enumerate(train_loaders_):
                print("Task", configs.train_tasks[j])
                
                print(f"Test loader {j} has {len(test_loader)} batches")
                if n_cls[configs.train_tasks[j]] != configs.n_cls:
                    pass
                else:
                    test(model, test_loader, loss_func, configs.n_cls, name=configs.train_tasks[j]) #, plot_feature="llm/task" + configs.train_tasks[j] + "test")
                    
                    
            
            print("="*10 + "test")
            for j, test_loader in enumerate(test_loaders):
                # test
                print("Task", configs.test_tasks[j])
                if  n_cls[configs.test_tasks[j]] != configs.n_cls:
                    # test(model, test_loader, loss_func, configs.n_cls, plot_feature="llm/task" + configs.test_tasks[j] + "test", plot_only=True)
                    pass
                else:
                    test(model, test_loader, loss_func, configs.n_cls, name=configs.test_tasks[j]) #, plot_feature="llm/task" + configs.test_tasks[j] + "test")
                    
        


def evaluate_RespLLM(configs):
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
    print("="*10 + "train set eval")
    for j, test_loader_S in enumerate(test_loaders_S):
        print("Task", configs.train_tasks[j])
        if n_cls[configs.test_tasks[j]] != configs.n_cls:
            pass
        else:
            test(model, test_loader_S, loss_func, configs.n_cls) #, plot_feature="llm/task" + configs.train_tasks[j] + "train")
    
    # # validation
    # print("="*10 + "validation")
    # validation_loss = 0
    # for j, val_loader in enumerate(val_loaders):
    #     validation_loss += test(model, val_loader, loss_func, configs.n_cls) #, plot_feature="llm/task" + configs.train_tasks[j] + "val")

    print("="*10 + "test")
    for j, test_loader in enumerate(test_loaders):
        # test
        print("Task", configs.test_tasks[j])
        if n_cls[configs.test_tasks[j]] != configs.n_cls:
            pass
        else:
            test(model, test_loader, loss_func, configs.n_cls)




if __name__ == "__main__":
    import os
    import wandb

    # Optionally set WANDB_API_KEY programmatically
    os.environ['WANDB_API_KEY'] = "redacted"

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
    # parser.add_argument("--test_sampler", default=False)
    parser.add_argument("--use_audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_audiolabel", action=argparse.BooleanOptionalAction, default=False)
    
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
    parser.add_argument("--method", type=str, default="universal") # universal, bayesshrink, sureshrink
    parser.add_argument("--wandb_name", type=str, default="test")

    args = parser.parse_args()
    print(args)

    args.train_tasks = args.train_tasks.split(",")
    args.test_tasks = args.test_tasks.split(",")

    if args.from_pretrain:
        evaluate_RespLLM(args)
    else:
        wandb.init(
            project=getattr(args, "wandb_project", "RespLLM_test"),
            name=getattr(args, "wandb_name", "test"),#"lowres_46_dynamic_"), #"Spread+MLP"
            config=vars(args)
        )
        train_RespLLM(args)

    send_telegram_message("Mô hình đã train xong.")