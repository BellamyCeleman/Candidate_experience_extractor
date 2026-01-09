import os
import torch

# === ПУТИ ===
BASE_DIR = os.path.dirname("XML_Roberta_neural_network_Anonimizator_finetune")
DATA_DIR = os.path.join(BASE_DIR)

SOURCE_FILE = os.path.join(DATA_DIR, "XML_Roberta_neural_network_Anonimizator_finetune/dataset.conll")
TRAIN_FILE = os.path.join(DATA_DIR, "XML_Roberta_neural_network_Anonimizator_finetune/train.txt")
VAL_FILE = os.path.join(DATA_DIR, "XML_Roberta_neural_network_Anonimizator_finetune/val.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "XML_Roberta_neural_network_Anonimizator_finetune/ner_model_output")

# === ПАРАМЕТРЫ МОДЕЛИ ===
MODEL_CONFIG = {
    "base_model": "xlm-roberta-base",     # Базовая модель для "new"
    "existing_model": "XML_Roberta_neural_network_Anonimizator_finetune/Model", # Путь для "continue" или "extend"
    "max_length": 512,
}

# === ПАРАМЕТРЫ ОБУЧЕНИЯ ===
TRAIN_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 20,
    "num_epochs": 1,
    "weight_decay": 0.01,
    "device": "cuda",
    "patience": 3
}

# === РЕЖИМ РАБОТЫ ===
# 'new'      - обучение с нуля
# 'continue' - продолжение обучения той же модели
# 'extend'   - добавление новых сущностей к старой модели
TRAINING_MODE = "new"