import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def get_base_model(model_name, label2id):
    """Загружает чистую модель с нуля."""
    print(f"🆕 Загрузка базовой модели: {model_name}")
    id2label = {v: k for k, v in label2id.items()}
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_existing_model(model_path):
    """Загружает уже обученную модель."""
    print(f"🔄 Загрузка существующей модели: {model_path}")
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def extend_model(model_path, new_label2id):
    """
    Расширяет существующую модель новыми классами.
    Копирует веса для совпадающих классов.
    """
    print(f"🔧 Расширение модели {model_path}...")

    # 1. Загружаем старую модель
    old_model = AutoModelForTokenClassification.from_pretrained(model_path)
    old_tokenizer = AutoTokenizer.from_pretrained(model_path)
    old_label2id = old_model.config.label2id

    # 2. Создаем новую модель с новой конфигурацией
    new_id2label = {v: k for k, v in new_label2id.items()}
    new_model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(new_label2id),
        id2label=new_id2label,
        label2id=new_label2id,
        ignore_mismatched_sizes=True
    )

    # 3. Копируем веса (Smart Weights Transfer)
    print("   ⚖️ Перенос весов...")
    with torch.no_grad():
        for label, old_id in old_label2id.items():
            if label in new_label2id:
                new_id = new_label2id[label]
                # Копируем классификатор (weights + bias)
                new_model.classifier.weight[new_id] = old_model.classifier.weight[old_id]
                new_model.classifier.bias[new_id] = old_model.classifier.bias[old_id]

    return new_model, old_tokenizer