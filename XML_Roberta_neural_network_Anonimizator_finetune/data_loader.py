import os
from pathlib import Path
from typing import List, Tuple, Dict
from datasets import Dataset
from sklearn.model_selection import train_test_split


def parse_conll(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """Читает CoNLL файл, возвращает (sentences, labels)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    sentences, labels = [], []
    tokens, tags = [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'): continue

            if not line:
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
                continue

            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) >= 2:
                tokens.append(parts[0])
                tags.append(parts[-1])

        if tokens:  # Добавляем последний буфер
            sentences.append(tokens)
            labels.append(tags)

    return sentences, labels


def split_and_save_data(source_path: str, train_path: str, val_path: str, split_ratio=0.2):
    """Делит исходный файл на train/val и сохраняет на диск."""
    print(f"📦 Разделение данных из {source_path}...")

    # Сначала парсим, чтобы делить по предложениям, а не по строкам
    sentences, labels = parse_conll(source_path)

    # Собираем обратно в текст для записи
    text_examples = []
    for sent, lab in zip(sentences, labels):
        lines = [f"{t} {l}" for t, l in zip(sent, lab)]
        text_examples.append("\n".join(lines))

    train_ex, val_ex = train_test_split(text_examples, test_size=split_ratio, random_state=42)

    Path(os.path.dirname(train_path)).mkdir(parents=True, exist_ok=True)

    with open(train_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(train_ex))
    with open(val_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(val_ex))

    print(f"✅ Созданы файлы: Train ({len(train_ex)}), Val ({len(val_ex)})")


def create_dataset(file_path: str, label2id: Dict[str, int] = None) -> Tuple[Dataset, Dict[str, int]]:
    """Создает HuggingFace Dataset и словарь меток."""
    sentences, labels = parse_conll(file_path)

    # Если маппинг не передан, создаем новый из данных
    if label2id is None:
        unique_tags = sorted(set(tag for seq in labels for tag in seq))
        label2id = {tag: i for i, tag in enumerate(unique_tags)}

    # Конвертация текстовых тегов в ID
    label_ids = []
    for seq in labels:
        # get('O', 0) защищает от неизвестных тегов
        ids = [label2id.get(t, label2id.get('O', 0)) for t in seq]
        label_ids.append(ids)

    dataset = Dataset.from_dict({"tokens": sentences, "ner_tags": label_ids})
    return dataset, label2id