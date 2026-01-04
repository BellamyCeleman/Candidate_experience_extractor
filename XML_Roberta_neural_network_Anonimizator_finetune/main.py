import os
import json
from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification, EarlyStoppingCallback

# Импорт наших модулей
import configs as cfg
import data_loader
import model_builder
import utils
import torch

def main():
    print("🚀 Запуск пайплайна обучения NER...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Используемое устройство: {device}")

    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Память: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    else:
        print("⚠️ ВНИМАНИЕ: Обучение будет идти на CPU. Это очень медленно!")

    # 1. Проверка и подготовка данных
    if not os.path.exists(cfg.TRAIN_FILE) or not os.path.exists(cfg.VAL_FILE):
        if not os.path.exists(cfg.SOURCE_FILE):
            raise FileNotFoundError(f"Нет исходного файла: {cfg.SOURCE_FILE}")
        data_loader.split_and_save_data(cfg.SOURCE_FILE, cfg.TRAIN_FILE, cfg.VAL_FILE)

    # 2. Создание Dataset объектов
    # Сначала создаем train, чтобы получить полный список меток
    train_ds, label2id = data_loader.create_dataset(cfg.TRAIN_FILE)
    # Используем тот же label2id для валидации
    val_ds, _ = data_loader.create_dataset(cfg.VAL_FILE, label2id=label2id)

    label_list = list(label2id.keys())
    print(f"🏷️ Классы ({len(label_list)}): {label_list}")

    # 3. Инициализация модели и токенайзера
    if cfg.TRAINING_MODE == "new":
        model, tokenizer = model_builder.get_base_model(cfg.MODEL_CONFIG["base_model"], label2id)
    elif cfg.TRAINING_MODE == "continue":
        model, tokenizer = model_builder.load_existing_model(cfg.MODEL_CONFIG["existing_model"])
    elif cfg.TRAINING_MODE == "extend":
        model, tokenizer = model_builder.extend_model(cfg.MODEL_CONFIG["existing_model"], label2id)
    else:
        raise ValueError(f"Неверный режим: {cfg.TRAINING_MODE}")

    # 4. Препроцессинг (токенизация)
    # Используем partial функцию или lambda для передачи токенайзера
    tokenize_fn = lambda x: utils.align_labels_with_tokens(x, tokenizer, cfg.MODEL_CONFIG["max_length"])

    print("⚙️ Токенизация...")
    train_encoded = train_ds.map(tokenize_fn, batched=True)
    val_encoded = val_ds.map(tokenize_fn, batched=True)

    # 5. Настройка аргументов обучения
    args = TrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        learning_rate=cfg.TRAIN_PARAMS["learning_rate"],
        per_device_train_batch_size=cfg.TRAIN_PARAMS["batch_size"],
        num_train_epochs=cfg.TRAIN_PARAMS["num_epochs"],
        weight_decay=cfg.TRAIN_PARAMS["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=os.path.join(cfg.OUTPUT_DIR, "logs"),
        fp16=True,  # Оставьте True, если карта серии RTX (20xx, 30xx, 40xx)
        no_cuda=False,
        dataloader_num_workers=0  # На Windows лучше оставить 0 или 1, чтобы избежать зависаний
    )

    # 6. Инициализация Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_encoded,
        eval_dataset=val_encoded,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=utils.compute_metrics_factory(label_list),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.TRAIN_PARAMS["patience"])]
    )

    # 7. Запуск
    print("\n🔥 Старт обучения...")
    trainer.train()

    # 8. Сохранение артефактов
    print(f"💾 Сохранение модели в {cfg.OUTPUT_DIR}")
    trainer.save_model(cfg.OUTPUT_DIR)
    tokenizer.save_pretrained(cfg.OUTPUT_DIR)

    # Сохраняем маппинг меток отдельно (полезно для инференса)
    with open(os.path.join(cfg.OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": {v: k for k, v in label2id.items()}}, f)

    # Финальная оценка
    metrics = trainer.evaluate()
    print(f"📊 Результаты: F1={metrics['eval_f1']:.4f}")


if __name__ == "__main__":
    main()