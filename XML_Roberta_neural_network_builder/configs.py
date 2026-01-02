import os
import torch

# === ПУТИ ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

SOURCE_FILE = os.path.join(DATA_DIR, "dataset.conll")
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
VAL_FILE = os.path.join(DATA_DIR, "val.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "ner_model_output")

# === ПАРАМЕТРЫ МОДЕЛИ ===
MODEL_CONFIG = {
    "base_model": "xlm-roberta-base",     # Базовая модель для "new"
    "existing_model": "ner_model_output", # Путь для "continue" или "extend"
    "max_length": 512,
}

# === ПАРАМЕТРЫ ОБУЧЕНИЯ ===
TRAIN_PARAMS = {
    "learning_rate": 2e-5,
    "batch_size": 16,
    "num_epochs": 10,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "patience": 3,   # Early stopping
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# === РЕЖИМ РАБОТЫ ===
# 'new'      - обучение с нуля
# 'continue' - продолжение обучения той же модели
# 'extend'   - добавление новых сущностей к старой модели
TRAINING_MODE = "extend"