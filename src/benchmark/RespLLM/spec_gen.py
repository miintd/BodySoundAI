import numpy as np
import collections
def get_dataloader(configs, task, sample=False, deft_seed=None):
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
    n_cls = collections.defaultdict(lambda:2, {"11": 5, "12": 5})

    dataset, label, modality = tasks_config[task]
    
    feature_dirs = {"covid19sounds": "feature/covid19sounds_eval", 
                    "coswara": "feature/coswara_eval/",
                    "coviduk": "feature/coviduk_eval/",
                    "kauh": "feature/kauh_eval/",
                    "icbhidisease": "feature/icbhidisease_eval/",
                    }
    
    pad_len_htsat = {"covid19sounds": 8.18, 
                    "coswara": 8.18,
                    "coviduk": 8.18,
                    "kauh": 8.18,
                    "icbhidisease": 8.18,
                    }
    feature_dir = feature_dirs[dataset]
    if task in ["S3", "S4"]:
        feature_dir = "feature/covid19sounds_eval/covid_eval/"
    elif task in ["S5", "S6"]:
        feature_dir = "feature/covid19sounds_eval/smoker_eval"
    
    if dataset in ["ssbpr", "copd", "kauh", "icbhidisease"]:
        suffix_dataset =  ".npy"
    elif dataset in ["covid19sounds", "coviduk"]:
        suffix_dataset = "_{}.npy".format(modality)
    elif dataset in ["coswara"]:
        suffix_dataset = "_{}_{}.npy".format(modality, label)
    elif dataset in ["coughvid"]:
        suffix_dataset = "_{}.npy".format(label)
    else:
        raise NotImplementedError
    

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
        else:
            label_dict = {"healthy": 0, "asthma": 1, "COPD": 1, "obstructive": 1}
            y_label = np.array([label_dict[y] for y in y_label])
    elif dataset == "coughvid":
        y_label = np.load(feature_dir + "label_{}.npy".format(label))
    else:
        y_label = np.load(feature_dir + "labels.npy")

    
    if dataset == "coswara":
        sound_dir_loc = np.load(
        feature_dir + "{}_aligned_filenames_{}_w_{}.npy".format(broad_modality, label, modality))
    elif dataset == "kauh":
        sound_dir_loc = np.load(feature_dir + "sound_dir_loc_subset.npy")
    else:
        sound_dir_loc = np.load(feature_dir + "sound_dir_loc" + suffix_dataset)

    
    from_audio = False

    spec_file_name = feature_dir + f"spectrogram_pad(1){str(int(pad_len_htsat[dataset]))}" + suffix_dataset if dataset != "icbhidisease" else feature_dir + f"segmented_spectrogram_pad(1){str(int(pad_len_htsat[dataset]))}" + suffix_dataset
    from src.util import get_split_signal_librosa
    x_data = []
    if dataset == "icbhidisease":
        y_segmented, y_set_segmented = [], []
        # x_metadata_segmented = []
        index_segmented = []
        y_set = np.load(feature_dir + "split.npy")
        for idx, audio_file in enumerate(sound_dir_loc):
            data = get_split_signal_librosa("", audio_file[:-4], spectrogram=True, input_sec=pad_len_htsat[dataset], trim_tail=False)
            if y_set[idx] == "train":
                # print([y_set[idx]], len(data))
                x_data.extend(data)
                y_segmented.extend([y_label[idx]] * len(data))
                y_set_segmented.extend([y_set[idx]] * len(data))
                # x_metadata_segmented.extend([x_metadata[idx]] * len(data))
                index_segmented.extend([idx] * len(data))
            else:
                # print([y_set[idx]])
                x_data.append(data[0])
                y_segmented.append(y_label[idx])
                y_set_segmented.append(y_set[idx])
                # x_metadata_segmented.append([x_metadata[idx]])
                index_segmented.append(idx)
        x_data = np.array(x_data)
        y_segmented = np.array(y_segmented)
        y_set_segmented = np.array(y_set_segmented)
        np.save(spec_file_name, x_data)
        np.save(feature_dir + f"segmented_split.npy", y_set_segmented)
        np.save(feature_dir + f"segmented_labels.npy", y_segmented)
        np.save(feature_dir + f"segmented_index.npy", index_segmented)
    else:
        for audio_file in sound_dir_loc:
            data = get_split_signal_librosa("", audio_file[:-4], spectrogram=True, input_sec=pad_len_htsat[dataset], trim_tail=False)[0]
            # print(data)
            x_data.append(data)
        x_data = np.array(x_data)
        np.save(spec_file_name, x_data)

if __name__ == "__main__":
    get_dataloader("S5")
    get_dataloader("S6") 