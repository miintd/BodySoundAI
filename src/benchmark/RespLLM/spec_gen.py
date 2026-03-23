import os
import numpy as np


TASKS_CONFIG = {
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

FEATURE_DIRS = {
    "covid19sounds": "feature/covid19sounds_eval/",
    "coswara": "feature/coswara_eval/",
    "coviduk": "feature/coviduk_eval/",
    "kauh": "feature/kauh_eval/",
    "icbhidisease": "feature/icbhidisease_eval/",
}

PAD_LEN_HTSAT = {
    "covid19sounds": 8.18,
    "coswara": 8.18,
    "coviduk": 8.18,
    "kauh": 8.18,
    "icbhidisease": 8.18,
}


def normalize_modality(modality: str) -> str:
    if modality in ["cough", "cough-shallow", "cough-heavy"]:
        return "cough"
    if modality in ["breath", "breathing-shallow", "breathing-deep", "exhalation"]:
        return "breath"
    if modality == "lung":
        return "lung"
    return modality


def get_feature_dir(task: str, dataset: str) -> str:
    feature_dir = FEATURE_DIRS[dataset]
    if task in ["S3", "S4"]:
        feature_dir = "feature/covid19sounds_eval/covid_eval/"
    elif task in ["S5", "S6"]:
        feature_dir = "feature/covid19sounds_eval/smoker_eval/"
    return feature_dir


def get_suffix_dataset(dataset: str, modality: str, label: str) -> str:
    if dataset in ["ssbpr", "copd", "kauh", "icbhidisease"]:
        return ".npy"
    elif dataset in ["covid19sounds", "coviduk"]:
        return f"_{modality}.npy"
    elif dataset == "coswara":
        return f"_{modality}_{label}.npy"
    elif dataset == "coughvid":
        return f"_{label}.npy"
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset}")


def get_sound_dir_loc(feature_dir: str, dataset: str, modality: str, label: str, suffix_dataset: str):
    if dataset == "coswara":
        broad_modality = modality.split("-")[0]
        return np.load(
            feature_dir + f"{broad_modality}_aligned_filenames_{label}_w_{modality}.npy",
            allow_pickle=True
        )
    elif dataset == "kauh":
        return np.load(feature_dir + "sound_dir_loc_subset.npy", allow_pickle=True)
    else:
        return np.load(feature_dir + "sound_dir_loc" + suffix_dataset, allow_pickle=True)


def build_spec_prefix(wavelet, wavelet_modality, wavelet_level, threshold_mode, method):
    if wavelet is None:
        return ""
    return f"{wavelet_modality}_{wavelet}_{wavelet_level}_{threshold_mode}_{method}_"


def build_spec_file_name(feature_dir, dataset, suffix_dataset, spec_prefix):
    base = f"spectrogram_pad{int(PAD_LEN_HTSAT[dataset])}"
    if dataset == "icbhidisease":
        base = f"segmented_spectrogram_pad{int(PAD_LEN_HTSAT[dataset])}"
    return feature_dir + spec_prefix + base + suffix_dataset


def get_baseline_file_name(feature_dir, dataset, suffix_dataset):
    base = f"spectrogram_pad{int(PAD_LEN_HTSAT[dataset])}"
    if dataset == "icbhidisease":
        base = f"segmented_spectrogram_pad{int(PAD_LEN_HTSAT[dataset])}"
    return feature_dir + base + suffix_dataset


def should_apply_wavelet(normalized_modality: str, wavelet_modality: str) -> bool:
    return wavelet_modality == "all" or normalized_modality == wavelet_modality


def generate_one_task(task, wavelet, wavelet_modality, wavelet_level, threshold_mode, method):
    from src.util import get_split_signal_librosa

    dataset, label, modality = TASKS_CONFIG[task]
    normalized_modality = normalize_modality(modality)

    feature_dir = get_feature_dir(task, dataset)
    suffix_dataset = get_suffix_dataset(dataset, modality, label)

    apply_wavelet = wavelet is not None and should_apply_wavelet(normalized_modality, wavelet_modality)

    effective_wavelet = wavelet if apply_wavelet else None
    effective_prefix = build_spec_prefix(
        effective_wavelet,
        wavelet_modality,
        wavelet_level,
        threshold_mode,
        method
    )

    spec_file_name = build_spec_file_name(feature_dir, dataset, suffix_dataset, effective_prefix)
    baseline_file_name = get_baseline_file_name(feature_dir, dataset, suffix_dataset)

    if os.path.exists(spec_file_name):
        print(f"[SKIP] {task} -> exists: {spec_file_name}")
        return

    # Nếu config không áp dụng lên task này, dùng baseline
    if effective_wavelet is None:
        if os.path.exists(baseline_file_name):
            print(f"[SKIP] {task} -> reuse baseline: {baseline_file_name}")
            return
        print(f"[RUN ] {task} -> creating baseline: {baseline_file_name}")
    else:
        print(
            f"[RUN ] {task} -> {wavelet_modality} | {wavelet} | "
            f"L{wavelet_level} | {method}"
        )

    sound_dir_loc = get_sound_dir_loc(feature_dir, dataset, modality, label, suffix_dataset)

    x_data = []

    if dataset == "icbhidisease":
        y_label = np.load(feature_dir + "labels.npy", allow_pickle=True)
        y_set = np.load(feature_dir + "split.npy", allow_pickle=True)

        y_segmented = []
        y_set_segmented = []
        index_segmented = []

        for idx, audio_file in enumerate(sound_dir_loc):
            data = get_split_signal_librosa(
                "",
                audio_file[:-4],
                spectrogram=True,
                input_sec=PAD_LEN_HTSAT[dataset],
                trim_tail=False,
                wavelet=effective_wavelet,
                wavelet_level=wavelet_level,
                threshold_mode=threshold_mode,
                method=method,
                normalized_modality=normalized_modality,
                wavelet_modality=wavelet_modality,
            )

            if y_set[idx] == "train":
                x_data.extend(data)
                y_segmented.extend([y_label[idx]] * len(data))
                y_set_segmented.extend([y_set[idx]] * len(data))
                index_segmented.extend([idx] * len(data))
            else:
                x_data.append(data[0])
                y_segmented.append(y_label[idx])
                y_set_segmented.append(y_set[idx])
                index_segmented.append(idx)

        x_data = np.array(x_data)
        y_segmented = np.array(y_segmented)
        y_set_segmented = np.array(y_set_segmented)
        index_segmented = np.array(index_segmented)

        save_path = spec_file_name if effective_wavelet is not None else baseline_file_name
        np.save(save_path, x_data)

        seg_split = feature_dir + "segmented_split.npy"
        seg_labels = feature_dir + "segmented_labels.npy"
        seg_index = feature_dir + "segmented_index.npy"

        if not os.path.exists(seg_split):
            np.save(seg_split, y_set_segmented)
        if not os.path.exists(seg_labels):
            np.save(seg_labels, y_segmented)
        if not os.path.exists(seg_index):
            np.save(seg_index, index_segmented)

        print(f"[DONE] {task} -> saved: {save_path}")
        return

    for audio_file in sound_dir_loc:
        data = get_split_signal_librosa(
            "",
            audio_file[:-4],
            spectrogram=True,
            input_sec=PAD_LEN_HTSAT[dataset],
            trim_tail=False,
            wavelet=effective_wavelet,
            wavelet_level=wavelet_level,
            threshold_mode=threshold_mode,
            method=method,
            normalized_modality=normalized_modality,
            wavelet_modality=wavelet_modality,
        )[0]
        x_data.append(data)

    x_data = np.array(x_data)
    save_path = spec_file_name if effective_wavelet is not None else baseline_file_name
    np.save(save_path, x_data)
    print(f"[DONE] {task} -> saved: {save_path}")


def get_tasks_for_wavelet_modality(wavelet_modality):
    if wavelet_modality == "cough":
        return ["S2", "S4", "S6", "T1", "T2"]
    if wavelet_modality == "breath":
        return ["S1", "S3", "S5", "T3", "T4"]
    if wavelet_modality == "lung":
        return ["S7", "T5", "T6"]
    if wavelet_modality == "all":
        return ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "T1", "T2", "T3", "T4", "T5", "T6"]
    raise ValueError(f"Invalid wavelet_modality: {wavelet_modality}")


def generate_baselines_once():
    print("\n" + "=" * 80)
    print("Generating baseline spectrograms once")
    print("=" * 80)

    for task in TASKS_CONFIG:
        try:
            generate_one_task(
                task=task,
                wavelet=None,
                wavelet_modality="all",
                wavelet_level=0,
                threshold_mode="soft",
                method="universal",
            )
        except Exception as e:
            print(f"[FAIL] baseline {task}: {e}")


def generate_ablations(selected_tasks=None):
    methods = ["universal", "bayesshrink"]

    ablations = {
        "cough": {
            "wavelets": ["db4", "sym4"],
            "levels": [4, 6],
        },
        "breath": {
            "wavelets": ["db8", "sym8"],
            "levels": [4, 6],
        },
        "lung": {
            "wavelets": ["coif2", "coif3"],
            "levels": [4, 6],
        },
        "all": {
            "wavelets": ["db4", "sym8", "coif3"],
            "levels": [4, 6],
        },
    }

    threshold_mode = "soft"

    for wavelet_modality, cfg in ablations.items():
        tasks = get_tasks_for_wavelet_modality(wavelet_modality)

        if selected_tasks is not None:
            tasks = [t for t in tasks if t in selected_tasks]

        if not tasks:
            continue

        for wavelet in cfg["wavelets"]:
            for level in cfg["levels"]:
                for method in methods:
                    print("\n" + "=" * 80)
                    print(
                        f"modality={wavelet_modality} | wavelet={wavelet} | "
                        f"level={level} | method={method}"
                    )
                    print(f"tasks={tasks}")
                    print("=" * 80)

                    for task in tasks:
                        try:
                            generate_one_task(
                                task=task,
                                wavelet=wavelet,
                                wavelet_modality=wavelet_modality,
                                wavelet_level=level,
                                threshold_mode=threshold_mode,
                                method=method,
                            )
                        except Exception as e:
                            print(f"[FAIL] {task}: {e}")


if __name__ == "__main__":
    # generate_baselines_once() 
    # generate_ablations()
    generate_ablations(selected_tasks=["S5","S6"])