"""
get_dataloader_wav: Load audio files as .wav instead of spectrogram .npy
"""
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.util import train_test_split_from_list
from src.benchmark.RespLLM.sampler import TrainCategoriesSampler
from torchmetrics import AUROC
import collections
import random
from src.benchmark.RespLLM.util import itr_merge, EarlyStopper, downsample_balanced_dataset, upsample_balanced_dataset, set_all_seed
from torchvggish.torchvggish.vggish_input import wavfile_to_examples


class AudioWavDataset(torch.utils.data.Dataset):
    """Load audio từ file .wav và tự động extract spectrogram"""
    
    def __init__(self, wav_paths, labels, modalities=None):
        self.wav_paths = wav_paths  # Array of file paths
        self.labels = labels
        self.modalities = modalities  # Array of modality strings
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        try:
            x = wavfile_to_examples(wav_path, return_tensor=True)  
            # (N,96,64)

            if x.shape[0] == 0:
                raise ValueError("Audio too short")
            x = x.mean(dim=0)  

        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            x = torch.zeros((1, 96, 64))  # đúng shape

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Return modality if available
        if self.modalities is not None:
            modality = self.modalities[idx]
            return x, label, modality
        else:
            return x, label


def get_dataloader_wav(configs, task, deft_seed=None, sample=False):
    tasks_config = {
        "S1": ("coviduk", "covid", "exhalation"),
        "S2": ("coviduk", "covid", "cough"),
        "S3": ("covid19sounds", "covid", "breath"),
        "S4": ("covid19sounds", "covid", "cough"),
        "S5": ("covid19sounds", "smoker", "breath"),
        "S6": ("covid19sounds", "smoker", "cough"),
        "S7": ("icbhidisease", "copd", "lung"),
        "T1": ("coswara", "covid", "cough-shallow"),
        "T2": ("coswara", "covid", "cough-heavy"),
        "T3": ("coswara", "covid", "breathing-shallow"),
        "T4": ("coswara", "covid", "breathing-deep"),
        "T5": ("kauh", "copd", "lung"),
        "T6": ("kauh", "asthma", "lung"),
    }
    n_cls = collections.defaultdict(lambda: 2, {"11": 5, "12": 5})

    dataset, label, modality = tasks_config[task]

    modality_map = {
        "cough": "cough",
        "cough-heavy": "cough",
        "cough-shallow": "cough",
        "breath": "breath",
        "breathing-deep": "breath",
        "breathing-shallow": "breath",
        "exhalation": "breath",
        "lung": "lung"
    }
    
    normalized_modality = modality_map[modality]

    feature_dirs = {
        "covid19sounds": "feature_/covid19sounds_eval/downsampled/", 
        "coswara": "feature_/coswara_eval/",
        "coviduk": "feature_/coviduk_eval/",
        "kauh": "feature_/kauh_eval/",
        "icbhidisease": "feature_/icbhidisease_eval/",
    }
    
    feature_dir = feature_dirs[dataset]
    if task in ["S3", "S4"]:
        feature_dir = "feature_/covid19sounds_eval/covid_eval/"
    elif task in ["S5", "S6"]:
        feature_dir = "feature_/covid19sounds_eval/smoker_eval/"
    
    # Suffix cho file labels
    if dataset in ["ssbpr", "copd", "kauh", "icbhidisease"]:
        suffix_dataset = ".npy"
    elif dataset in ["covid19sounds", "coviduk"]:
        suffix_dataset = "_{}.npy".format(modality)
    elif dataset in ["coswara"]:
        suffix_dataset = "_{}_{}.npy".format(modality, label)
    elif dataset in ["coughvid"]:
        suffix_dataset = "_{}.npy".format(label)
    else:
        raise NotImplementedError

    # Load labels (same as original)
    if dataset == "coviduk":
        y_label = np.load(feature_dir + f"label_{modality}.npy")
    elif dataset == "coswara":
        broad_modality = modality.split("-")[0]
        y_label = np.load(feature_dir + "{}_aligned_{}_label_{}.npy".format(broad_modality, label, modality))
    elif dataset == "kauh":
        y_label = np.load(feature_dir + "labels_both.npy")
        if label == "copd":
            label_dict = {"healthy": 0, "asthma": 2, "COPD": 1, "obstructive": 2}
            y_label = np.array([label_dict[y] for y in y_label])
        elif label == "asthma":
            label_dict = {"healthy": 0, "asthma": 1, "COPD": 2, "obstructive": 2}
            y_label = np.array([label_dict[y] for y in y_label])
    elif dataset == "coughvid":
        y_label = np.load(feature_dir + "label_{}.npy".format(label))
    else:
        y_label = np.load(feature_dir + "labels.npy")
    
    seed = 42
    
    # LOAD AUDIO FILE PATHS từ sound_dir_loc.npy
    # (thay vì load spectrograms từ vggish_spectrogram_pad*.npy)
    if dataset == "coswara":
        sound_dir_loc = np.load(feature_dir + "{}_aligned_filenames_{}_w_{}.npy".format(
            modality.split("-")[0], label, modality))
    elif dataset == "kauh":
        sound_dir_loc = np.load(feature_dir + "sound_dir_loc_subset.npy")
    else:
        sound_dir_loc = np.load(feature_dir + "sound_dir_loc" + suffix_dataset)
    
    print(f"Loading {len(sound_dir_loc)} audio files for task {task}")
    print(collections.Counter(y_label))

    x_metadata = np.array(["" for x in range(len(y_label))])

    # Split data (same logic as original)
    if dataset == "covid19sounds":
        y_set = np.load(feature_dir + "data_split.npy")
        
        if task in ["S3", "S4"]:
            sound_dir_loc_train = sound_dir_loc[y_set == "train"]
            x_metadata_train = x_metadata[y_set == "train"]
            y_label_train = y_label[y_set == "train"]
            
            sound_dir_loc_vad = sound_dir_loc[y_set == "validation"]
            x_metadata_vad = x_metadata[y_set == "validation"]
            y_label_vad = y_label[y_set == "validation"]

            sound_dir_loc_test = sound_dir_loc[y_set == "test"]
            x_metadata_test = x_metadata[y_set == "test"]
            y_label_test = y_label[y_set == "test"]
        else:
            sound_dir_loc_train = sound_dir_loc[y_set == 0]
            x_metadata_train = x_metadata[y_set == 0]
            y_label_train = y_label[y_set == 0]
            
            sound_dir_loc_vad = sound_dir_loc[y_set == 1]
            x_metadata_vad = x_metadata[y_set == 1]
            y_label_vad = y_label[y_set == 1]

            sound_dir_loc_test = sound_dir_loc[y_set == 2]
            x_metadata_test = x_metadata[y_set == 2]
            y_label_test = y_label[y_set == 2]
    
    elif dataset == "coviduk":
        y_set = np.load(feature_dir + "split_{}.npy".format(modality))
        
        sound_dir_loc_train = sound_dir_loc[y_set == "train"]
        x_metadata_train = x_metadata[y_set == "train"]
        y_label_train = y_label[y_set == "train"]
        
        sound_dir_loc_vad = sound_dir_loc[y_set == "val"]
        x_metadata_vad = x_metadata[y_set == "val"] 
        y_label_vad = y_label[y_set == "val"]

        sound_dir_loc_test = sound_dir_loc[y_set == "test"]
        x_metadata_test = x_metadata[y_set == "test"]
        y_label_test = y_label[y_set == "test"]
    
    elif dataset == "kauh":
        y_set = np.load(feature_dir + "train_test_split.npy")
        if label in ["copd", "asthma"]:
            mask = (y_label == 0) | (y_label == 1)
            y_label = y_label[mask]
            y_set = y_set[mask]
            sound_dir_loc = sound_dir_loc[mask]
        
        sound_dir_loc_train, sound_dir_loc_test, _, _ = train_test_split_from_list(
            sound_dir_loc, y_label, y_set
        )
        x_metadata_train, _, y_label_train, y_label_test = train_test_split_from_list(
            x_metadata, y_label, y_set
        )
        sound_dir_loc_train, sound_dir_loc_vad, y_label_train, y_label_vad = train_test_split(
            sound_dir_loc_train, y_label_train, test_size=0.1, 
            random_state=1337, stratify=y_label_train
        )
    
    elif dataset == "coswara":
        # _x_train, sound_dir_loc_test, _x_metadata_train, _, _y_train, y_label_test = train_test_split(
        #     sound_dir_loc, x_metadata, y_label, test_size=0.2, random_state=seed, stratify=y_label
        # )
        # sound_dir_loc_train, sound_dir_loc_vad, y_label_train, y_label_vad = train_test_split(
        #     _x_train, _y_train, test_size=0.2, 
        #     random_state=seed, stratify=_y_train
        # )
        if label == "covid":
            set_all_seed(seed)
            # symptoms = np.array([1 if 'following respiratory symptoms' in m else 0 for m in x_metadata])
            symptoms = np.load(feature_dir + f"symptom" + suffix_dataset)

            group1_indices = np.where((y_label == 0) & (symptoms == 1))[0]
            group2_indices = np.where((y_label == 0) & (symptoms == 0))[0]
            group3_indices = np.where((y_label == 1) & (symptoms == 1))[0]
            group4_indices = np.where((y_label == 1) & (symptoms == 0))[0]
            random.seed(seed)

            test_size = np.min([len(group) for group in [group1_indices, group2_indices, group3_indices, group4_indices]]) - (configs.meta_val_shot // 2)

            def sample_indices(group_indices, test_size):
                print(f"sampling {test_size} from", len(group_indices))
                test_sample_indices = np.random.choice(group_indices, size=test_size, replace=False)
                remaining_indices = np.setdiff1d(group_indices, test_sample_indices)
                return test_sample_indices, remaining_indices
        
            # Step 2: Sample 30 indices for each group for the test set
            group1_indices_test, group1_indices_train = sample_indices(group1_indices, test_size)
            group2_indices_test, group2_indices_train = sample_indices(group2_indices, test_size)
            group3_indices_test, group3_indices_train = sample_indices(group3_indices, test_size)
            group4_indices_test, group4_indices_train = sample_indices(group4_indices, test_size)

            # Combine test and training indices
            indices_test = np.concatenate([group1_indices_test, group2_indices_test, group3_indices_test, group4_indices_test])
            indices_train = np.concatenate([group1_indices_train, group2_indices_train, group3_indices_train, group4_indices_train])

            print("train")
            for indices_array in [group1_indices_train, group2_indices_train, group3_indices_train, group4_indices_train]:
                print(len(indices_array), end=";")
            print("\ntest")
            for indices_array in[group1_indices_test, group2_indices_test, group3_indices_test, group4_indices_test]:
                print(len(indices_array), end=";")
            print()
            # Step 3: Use the sampled indices to get the test and training data
            sound_dir_loc_test = sound_dir_loc[indices_test]
            x_metadata_test = x_metadata[indices_test]
            y_label_test = y_label[indices_test]
            symptoms_test = symptoms[indices_test]

            sound_dir_loc_train = sound_dir_loc[indices_train]
            x_metadata_train = x_metadata[indices_train]
            y_label_train = y_label[indices_train]
            symptoms_train = symptoms[indices_train]

            sound_dir_loc_vad, x_metadata_vad, y_label_vad = sound_dir_loc_train, x_metadata_train, y_label_train 

            group_idxs = []
            for i in range(len(sound_dir_loc_train)):
                y = y_label_train[i]
                m = symptoms_train[i]
                if y == 0 and m == 1:
                    group = 1
                if y == 0 and m == 0:
                    group = 2
                if y == 1 and m == 1:
                    group = 3
                if y == 1 and m == 0:
                    group = 4
                group_idxs.append(group)
        
            group_idxs = np.array(group_idxs)
        
        else:
            # smoker
            _x_train, sound_dir_loc_test, _x_metadata_train, _, _y_train, y_label_test = train_test_split(
                sound_dir_loc, x_metadata, y_label, test_size=0.2, random_state=seed, stratify=y_label
            )
            sound_dir_loc_train, sound_dir_loc_vad, y_label_train, y_label_vad = train_test_split(
                _x_train, _y_train, test_size=0.2, 
                random_state=seed, stratify=_y_train
            )
    else: # icbhi 
        y_set = np.load(feature_dir + "split.npy")
        mask = (y_label == "Healthy") | (y_label == "COPD")
        y_label = y_label[mask]
        y_set = y_set[mask]
        sound_dir_loc = sound_dir_loc[mask]
        label_dict = {"Healthy": 0, "COPD": 1}
        y_label = np.array([label_dict[y] for y in y_label])

        sound_dir_loc_train, sound_dir_loc_test, y_label_train, y_label_test = train_test_split_from_list(sound_dir_loc, y_label, y_set)
        x_metadata_train, x_metadata_test, y_label_train, y_label_test = train_test_split_from_list(x_metadata, y_label, y_set)

        sound_dir_loc_train, sound_dir_loc_vad, x_metadata_train, x_metadata_vad, y_label_train, y_label_vad = train_test_split(
                sound_dir_loc_train, x_metadata_train, y_label_train, test_size=0.2, 
                random_state=1337, stratify=y_label_train
            )

    if task in ["S5", "S6"]:
        sound_dir_loc_train, x_metadata_train, y_label_train = downsample_balanced_dataset(sound_dir_loc_train, x_metadata_train, y_label_train)
    if task in ["S7"]:
        sound_dir_loc_train, x_metadata_train, y_label_train = upsample_balanced_dataset(sound_dir_loc_train, x_metadata_train, y_label_train)

    x_modality_train = np.array([normalized_modality] * len(sound_dir_loc_train))
    x_modality_vad   = np.array([normalized_modality] * len(sound_dir_loc_vad))
    x_modality_test  = np.array([normalized_modality] * len(sound_dir_loc_test))

    train_data_percentage = configs.train_pct
    if not sample and train_data_percentage < 1:
        sound_dir_loc_train, _, y_label_train, _, x_metadata_train, _ = train_test_split(
            sound_dir_loc_train, y_label_train, x_metadata_train, test_size=1 - train_data_percentage, random_state=seed, stratify=y_label_train
        )
        x_modality_train = np.array([normalized_modality] * len(sound_dir_loc_train))

    print(collections.Counter(y_label_train))
    min_train_cls = min(collections.Counter(y_label_train).values())
    print(collections.Counter(y_label_vad))
    print(collections.Counter(y_label_test))
    min_test_cls = min(collections.Counter(y_label_test).values())
   
    # Create datasets using WAV files
    train_data = AudioWavDataset(sound_dir_loc_train, y_label_train, x_modality_train)
    test_data = AudioWavDataset(sound_dir_loc_test, y_label_test, x_modality_test)
    val_data = AudioWavDataset(sound_dir_loc_vad, y_label_vad, x_modality_vad)


    # Create dataloaders
    train_loader = DataLoader(
        train_data, batch_size=configs.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=configs.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_data, batch_size=configs.batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader, test_loader


def test(model, test_loader, loss_func, n_cls, return_auc=False, verbose=True):
    """Test function - evaluate model on test set"""
    total_loss = []
    test_step_outputs = []
    features = []
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Handle both (x, y) and (x, y, modality) batch formats
            if len(batch) == 3:
                x1, y, x_modality = batch
            else:
                x1, y = batch
                x_modality = None
                
            x1 = x1.to(device)
            y = y.to(device)
            
            # Call model with modality if available and model supports it
            if x_modality is not None and hasattr(model, 'modal_embs') and model.modal_embs is not None:
                y_hat = model(x1, x_modality)
            else:
                y_hat = model(x1)

            loss = loss_func(y_hat, y)
            total_loss.append(loss.item())

            _, predicted = torch.max(y_hat, 1)
            probabilities = F.softmax(y_hat, dim=1)
            test_step_outputs.append((y.detach().cpu().numpy(), predicted.detach().cpu().numpy(), probabilities.detach().cpu().numpy() ))
    
    all_outputs = test_step_outputs
    y = np.concatenate([output[0] for output in all_outputs])
    
    total_loss = np.average(total_loss)
    if verbose:
        print("loss", total_loss)

    predicted = np.concatenate([output[1] for output in all_outputs])
    probs = np.concatenate([output[2] for output in all_outputs])

    acc = np.mean(predicted == y)

    auroc = AUROC(task="multiclass", num_classes=n_cls)
    auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))

    cm = confusion_matrix(y, predicted)
    TN, FP, FN, TP = cm.ravel()
    
    if verbose:
        print("loss", total_loss)
        print("acc", acc)
        print("auc", auc)
        print(f"TP: {TP}")
        print(f"TN: {TN}")
        print(f"FP: {FP}")
        print(f"FN: {FN}")
    if return_auc:
        return acc, auc
    return total_loss / (i + 1)
