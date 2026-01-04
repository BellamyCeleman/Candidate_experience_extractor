"""
Тестирование обученной NER модели
"""

from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch


def load_model(model_path: str):
    """Загружает модель и токенизатор"""
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    # GPU если доступен
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"✅ Модель загружена: {model_path}")
    print(f"   Device: {device}")
    print(f"   Labels: {list(model.config.id2label.values())}")

    return model, tokenizer, device


def predict(text: str, model, tokenizer, device) -> list[dict]:
    """Извлекает сущности из текста"""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [model.config.id2label[p.item()] for p in predictions[0]]

    entities = []
    current_entity = None
    current_tokens = []

    for token, label in zip(tokens, labels):
        if token in ["<s>", "</s>", "<pad>"]:
            continue

        if label.startswith("B-"):
            # Сохраняем предыдущую сущность
            if current_entity:
                text_value = tokenizer.convert_tokens_to_string(current_tokens).strip()
                if text_value:
                    entities.append({"type": current_entity, "text": text_value})

            # Начинаем новую
            current_entity = label[2:]
            current_tokens = [token]

        elif label.startswith("I-") and current_entity == label[2:]:  # ← ИСПРАВЛЕНО
            # Продолжаем ТОЛЬКО если тип совпадает
            current_tokens.append(token)

        else:  # "O" или несовпадение типа
            if current_entity:
                text_value = tokenizer.convert_tokens_to_string(current_tokens).strip()
                if text_value:
                    entities.append({"type": current_entity, "text": text_value})
                current_entity = None
                current_tokens = []

    # Последняя сущность
    if current_entity:
        text_value = tokenizer.convert_tokens_to_string(current_tokens).strip()
        if text_value:
            entities.append({"type": current_entity, "text": text_value})

    return entities


def print_entities(entities: list[dict]):
    """Красиво выводит сущности"""
    if not entities:
        print("   Сущности не найдены")
        return

    # Группируем по типу
    by_type = {}
    for e in entities:
        t = e["type"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(e["text"])

    for entity_type, values in sorted(by_type.items()):
        print(f"   {entity_type}: {', '.join(values)}")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    # Путь к обученной модели
    MODEL_PATH = "XML_Roberta_neural_network_Anonimizator_finetune/ner_model_output"

    # Загружаем модель
    model, tokenizer, device = load_model(MODEL_PATH)

    # Тестовые примеры
    test_texts = [
        "Иванов Иван Петрович - Senior Python Developer в компании Google, Киев. Навыки: Python, Django, PostgreSQL. Опыт: 2020-2024.",

        "Bezkorovainy Mykyta worked as Angular Developer at SmartFox Pro from March 2022. Skills: Angular, TypeScript, RxJS.",

        "Шевченко Тарас, Junior Data Scientist, SoftServe, Львів. Python, TensorFlow, Pandas. 01/2023 - present.",

        "SQL Developer. Python, Django, Flask",
    ]

    print("\n" + "=" * 60)
    print("🧪 ТЕСТИРОВАНИЕ NER МОДЕЛИ")
    print("=" * 60)

    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Текст {i}:")
        print(f"   {text[:80]}..." if len(text) > 80 else f"   {text}")

        entities = predict(text, model, tokenizer, device)

        print(f"\n🏷️ Найденные сущности:")
        print_entities(entities)
        print("-" * 60)

    # Интерактивный режим
    print("\n💬 Интерактивный режим (введите 'exit' для выхода):")
    while True:
        try:
            user_text = input("\nВведите текст: ").strip()
            if user_text.lower() == "exit":
                break
            if not user_text:
                continue

            entities = predict(user_text, model, tokenizer, device)
            print("\n🏷️ Сущности:")
            print_entities(entities)

        except KeyboardInterrupt:
            break

    print("\n👋 Готово!")