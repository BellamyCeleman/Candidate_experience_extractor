"""
NER экстрактор с позициями сущностей в тексте
"""

from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import json
from dataclasses import dataclass, asdict
from typing import Optional

from .config import XLM_RoBerta_entities_extractor_config


@dataclass
class Entity:
    """Найденная сущность"""
    type: str
    text: str
    start: int
    end: int


class XLM_RoBerta_entities_extractor:
    """Извлекает именованные сущности с их позициями в тексте"""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Args:
            model_path: Путь к модели (если None - берётся из конфига)
            device: Устройство для инференса ("cuda", "cpu" или None для автовыбора)
        """
        if model_path is None:
            config = XLM_RoBerta_entities_extractor_config()
            model_path = config.MODEL_PATH

        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

    def extract(self, text: str) -> list[Entity]:
        """
        Извлекает сущности из текста.

        Args:
            text: Входной текст

        Returns:
            Список Entity с типом, текстом и позициями (start, end)
        """
        if not text or not text.strip():
            return []

        # Токенизация с offset_mapping для получения позиций
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )

        offset_mapping = inputs.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Предсказание
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()

        # Собираем сущности
        entities = []
        current_entity_type = None
        current_start = None
        current_end = None

        for idx, (pred_id, (start, end)) in enumerate(zip(predictions, offset_mapping)):
            # Пропускаем специальные токены
            if start == 0 and end == 0:
                continue

            label = self.model.config.id2label[pred_id]

            if label.startswith("B-"):
                # Сохраняем предыдущую сущность
                if current_entity_type is not None:
                    entity_text = text[current_start:current_end]
                    if entity_text.strip():
                        entities.append(Entity(
                            type=current_entity_type,
                            text=entity_text.strip(),
                            start=current_start,
                            end=current_end
                        ))

                # Начинаем новую сущность
                current_entity_type = label[2:]
                current_start = start
                current_end = end

            elif label.startswith("I-") and current_entity_type == label[2:]:
                # Продолжаем текущую сущность (только если тип совпадает)
                current_end = end

            else:  # "O" или несовпадение типа
                if current_entity_type is not None:
                    entity_text = text[current_start:current_end]
                    if entity_text.strip():
                        entities.append(Entity(
                            type=current_entity_type,
                            text=entity_text.strip(),
                            start=current_start,
                            end=current_end
                        ))
                    current_entity_type = None
                    current_start = None
                    current_end = None

        # Последняя сущность
        if current_entity_type is not None:
            entity_text = text[current_start:current_end]
            if entity_text.strip():
                entities.append(Entity(
                    type=current_entity_type,
                    text=entity_text.strip(),
                    start=current_start,
                    end=current_end
                ))

        return entities

    def extract_to_json(self, text: str) -> dict:
        """
        Извлекает сущности и возвращает JSON-совместимый словарь.

        Returns:
            {
                "text": "оригинальный текст",
                "entities": [
                    {"type": "PERSON", "text": "Иванов Иван", "start": 0, "end": 11},
                    ...
                ],
                "entities_by_type": {
                    "PERSON": ["Иванов Иван"],
                    "ORG": ["Google"],
                    ...
                }
            }
        """
        entities = self.extract(text)

        # Группируем по типу
        by_type: dict[str, list[str]] = {}
        for e in entities:
            if e.type not in by_type:
                by_type[e.type] = []
            if e.text not in by_type[e.type]:  # Избегаем дубликатов
                by_type[e.type].append(e.text)

        return {
            "text": text,
            "entities": [asdict(e) for e in entities],
            "entities_by_type": by_type
        }

    def anonymize(
        self,
        text: str,
        placeholder_format: str = "[{type}]",
        entity_types: Optional[list[str]] = None
    ) -> dict:
        """
        Заменяет сущности в тексте на плейсхолдеры.

        Args:
            text: Исходный текст
            placeholder_format: Формат плейсхолдера:
                - "[{type}]" -> [PERSON], [ORG], [DATE]
                - "[REDACTED]" -> всё заменяется на [REDACTED]
                - "***" -> всё заменяется на ***
            entity_types: Список типов для замены (None = все типы)

        Returns:
            {
                "original_text": "...",
                "anonymized_text": "...",
                "replacements": [
                    {"type": ..., "original": ..., "replacement": ..., "start": ..., "end": ...}
                ]
            }
        """
        entities = self.extract(text)

        # Фильтруем по типам если указано
        if entity_types:
            entities = [e for e in entities if e.type in entity_types]

        if not entities:
            return {
                "original_text": text,
                "anonymized_text": text,
                "replacements": []
            }

        # Сортируем по позиции в обратном порядке (с конца),
        # чтобы замены не сбивали индексы
        entities_sorted = sorted(entities, key=lambda e: e.start, reverse=True)

        anonymized = text
        replacements = []

        for entity in entities_sorted:
            # Формируем плейсхолдер
            if "{type}" in placeholder_format:
                placeholder = placeholder_format.format(type=entity.type)
            else:
                placeholder = placeholder_format

            # Заменяем
            anonymized = anonymized[:entity.start] + placeholder + anonymized[entity.end:]

            replacements.append({
                "type": entity.type,
                "original": entity.text,
                "replacement": placeholder,
                "start": entity.start,
                "end": entity.end
            })

        # Разворачиваем для хронологического порядка
        replacements.reverse()

        return {
            "original_text": text,
            "anonymized_text": anonymized,
            "replacements": replacements
        }


# ============================================
# Удобные функции для быстрого использования
# ============================================

_extractor: Optional[XLM_RoBerta_entities_extractor] = None


def init_extractor(model_path: Optional[str] = None, device: Optional[str] = None):
    """Инициализирует глобальный экстрактор"""
    global _extractor
    _extractor = XLM_RoBerta_entities_extractor(model_path, device)


def extract_entities(text: str) -> dict:
    """
    Извлекает сущности из текста.

    Args:
        text: Текст для анализа

    Returns:
        JSON-совместимый словарь с сущностями и их позициями
    """
    if _extractor is None:
        raise RuntimeError("Сначала вызовите init_extractor()")
    return _extractor.extract_to_json(text)


def anonymize_text(
    text: str,
    placeholder_format: str = "[{type}]",
    entity_types: Optional[list[str]] = None
) -> dict:
    """
    Анонимизирует текст.

    Args:
        text: Текст для анонимизации
        placeholder_format: Формат замены
        entity_types: Типы сущностей для замены (None = все)

    Returns:
        Словарь с original_text, anonymized_text, replacements
    """
    if _extractor is None:
        raise RuntimeError("Сначала вызовите init_extractor()")
    return _extractor.anonymize(text, placeholder_format, entity_types)


# ============================================
# MAIN - демонстрация
# ============================================

if __name__ == "__main__":
    # Путь берётся из конфига автоматически
    extractor = XLM_RoBerta_entities_extractor()

    test_text = "Іванов Іван Петрович - Senior Python Developer в компанії Google, Київ. Навички: Python, Django, PostgreSQL. Досвід: 2020-2024."

    print("=" * 60)
    print("🧪 ТЕСТ NER ЭКСТРАКТОРА")
    print("=" * 60)

    # Извлечение сущностей
    print("\n📋 Сущности:")
    entities = extractor.extract(test_text)
    for e in entities:
        print(f"   [{e.start}:{e.end}] {e.type}: '{e.text}'")

    # JSON формат
    print("\n📄 JSON:")
    result = extractor.extract_to_json(test_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Анонимизация
    print("\n" + "=" * 60)
    print("🔒 АНОНИМИЗАЦИЯ")
    print("=" * 60)

    print(f"\n📝 Оригинал:\n   {test_text}")

    # Стандартный формат
    anon = extractor.anonymize(test_text)
    print(f"\n🔒 [{{type}}]:\n   {anon['anonymized_text']}")

    # Единый плейсхолдер
    anon2 = extractor.anonymize(test_text, placeholder_format="[REDACTED]")
    print(f"\n🔒 [REDACTED]:\n   {anon2['anonymized_text']}")

    # Только PERSON
    anon3 = extractor.anonymize(test_text, entity_types=["PER"])
    print(f"\n🔒 Только PERSON:\n   {anon3['anonymized_text']}")