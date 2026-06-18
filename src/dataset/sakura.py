import os
import json
import numpy as np
import datasets
import librosa
from tqdm import tqdm
from scipy.io import wavfile
from torch.utils.data import Dataset

from src import Define


class _BaseSakuraDataset(Dataset):
    HF_DATASET: str = ""
    LABELS: list[str] = []
    SUBJECT: str = ""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir or os.path.join(Define.CACHE_DIR, "SAKURA", self.SUBJECT)
        self.data_info_path = os.path.join(self.cache_dir, "data_info.json")

        if not os.path.isfile(self.data_info_path):
            self._parse()

        with open(self.data_info_path, "r", encoding="utf-8") as f:
            self.info = json.load(f)

    def _parse(self):
        src = datasets.load_dataset(self.HF_DATASET, split="test")
        os.makedirs(os.path.join(self.cache_dir, "wav"), exist_ok=True)

        res = []
        for instance in tqdm(src, desc=f"Caching SAKURA {self.SUBJECT}"):
            wav_path = os.path.join(self.cache_dir, "wav", instance["file"])
            if not os.path.isfile(wav_path):
                wav = librosa.resample(
                    instance["audio"]["array"],
                    orig_sr=instance["audio"]["sampling_rate"],
                    target_sr=16000,
                )
                wavfile.write(wav_path, 16000, (wav * 32767).astype(np.int16))
            res.append({
                "audio_path": os.path.join("wav", instance["file"]),
                "label": instance["attribute_label"],
            })

        with open(self.data_info_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=4)

    def __len__(self) -> int:
        return len(self.info)

    def __getitem__(self, idx: int) -> dict:
        item = self.info[idx]
        audio_path = os.path.join(self.cache_dir, item["audio_path"])
        audio, _ = librosa.load(audio_path, sr=16000)
        return {
            "id": item["audio_path"],
            "audio": audio,        # np.ndarray, float32, 16kHz
            "label": item["label"],
        }


class EmotionSAKURA(_BaseSakuraDataset):
    HF_DATASET = "SLLM-multi-hop/EmotionQA"
    LABELS = ["disgust", "sad", "angry", "fear", "happy"]
    SUBJECT = "emotion"


class AnimalSAKURA(_BaseSakuraDataset):
    HF_DATASET = "SLLM-multi-hop/AnimalQA"
    LABELS = ["cat", "cow", "crow", "dog", "frog", "hen", "pig", "rooster", "sheep"]
    SUBJECT = "animal"


class GenderSAKURA(_BaseSakuraDataset):
    HF_DATASET = "SLLM-multi-hop/GenderQA"
    LABELS = ["female", "male"]
    SUBJECT = "gender"


class LanguageSAKURA(_BaseSakuraDataset):
    HF_DATASET = "SLLM-multi-hop/LanguageQA"
    LABELS = ["Chinese", "English", "French", "German", "Italian", "Japanese", "Korean", "Spanish"]
    SUBJECT = "language"


SAKURA_TASKS: dict[str, type[_BaseSakuraDataset]] = {
    "emotion": EmotionSAKURA,
    "animal": AnimalSAKURA,
    "gender": GenderSAKURA,
    "language": LanguageSAKURA,
}
