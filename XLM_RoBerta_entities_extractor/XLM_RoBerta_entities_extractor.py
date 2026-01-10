"""
NER экстрактор с позициями сущностей в тексте
"""

from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import json
from dataclasses import dataclass, asdict
from typing import Optional

from RFC_logging_system.LoggerFactory import get_logger

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
        self.logger = get_logger("XLM_RoBerta_entities_extractor")
        
        if model_path is None:
            config = XLM_RoBerta_entities_extractor_config()
            model_path = config.MODEL_PATH
            self.logger.debug(f"Using model path from config: {model_path}")
        else:
            self.logger.debug(f"Using provided model path: {model_path}")

        self.logger.info(f"Loading model from: {model_path}")
        try:
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model.eval()
            self.logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {str(e)}")
            raise

        if device:
            self.device = torch.device(device)
            self.logger.debug(f"Using specified device: {device}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.debug(f"Auto-detected device: {self.device}")

        self.model.to(self.device)
        self.logger.info(f"Model moved to device: {self.device}")

    def extract(self, text: str) -> list[Entity]:
        """
        Извлекает сущности из текста.

        Args:
            text: Входной текст

        Returns:
            Список Entity с типом, текстом и позициями (start, end)
        """
        self.logger.debug(f"Starting entity extraction for text: {text[:100]}...")
        
        if not text or not text.strip():
            self.logger.warning("Empty text provided for entity extraction")
            return []

        try:
            # Токенизация с offset_mapping для получения позиций
            self.logger.debug("Tokenizing input text")
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
            self.logger.debug("Running model inference")
            with torch.no_grad():
                outputs = self.model(**inputs)

            predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
            self.logger.debug(f"Model predictions generated, total tokens: {len(predictions)}")

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

            self.logger.info(f"Entity extraction completed. Found {len(entities)} entities")
            for entity in entities:
                self.logger.debug(f"Found entity: [{entity.start}:{entity.end}] {entity.type}: '{entity.text}'")
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Error during entity extraction: {str(e)}")
            raise

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
        self.logger.debug("Starting JSON extraction process")
        
        try:
            entities = self.extract(text)

            # Группируем по типу
            by_type: dict[str, list[str]] = {}
            for e in entities:
                if e.type not in by_type:
                    by_type[e.type] = []
                if e.text not in by_type[e.type]:  # Избегаем дубликатов
                    by_type[e.type].append(e.text)

            result = {
                "text": text,
                "entities": [asdict(e) for e in entities],
                "entities_by_type": by_type
            }
            
            self.logger.debug(f"JSON extraction completed. Total entities: {len(entities)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during JSON extraction: {str(e)}")
            raise

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
        self.logger.debug(f"Starting text anonymization with format: {placeholder_format}")
        
        try:
            entities = self.extract(text)

            # Фильтруем по типам если указано
            if entity_types:
                self.logger.debug(f"Filtering entities by types: {entity_types}")
                entities = [e for e in entities if e.type in entity_types]

            if not entities:
                self.logger.debug("No entities found for anonymization")
                return {
                    "original_text": text,
                    "anonymized_text": text,
                    "replacements": []
                }

            # Сортируем по позиции в обратном порядке (с конца),
            # чтобы замены не сбивали индексы
            entities_sorted = sorted(entities, key=lambda e: e.start, reverse=True)
            self.logger.debug(f"Found {len(entities)} entities to anonymize")

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

                self.logger.debug(f"Replaced entity: '{entity.text}' -> '{placeholder}' at [{entity.start}:{entity.end}]")

            # Разворачиваем для хронологического порядка
            replacements.reverse()

            self.logger.info(f"Anonymization completed. {len(replacements)} replacements made")
            return {
                "original_text": text,
                "anonymized_text": anonymized,
                "replacements": replacements
            }
            
        except Exception as e:
            self.logger.error(f"Error during text anonymization: {str(e)}")
            raise


# ============================================
# Удобные функции для быстрого использования
# ============================================

_extractor: Optional[XLM_RoBerta_entities_extractor] = None


def init_extractor(model_path: Optional[str] = None, device: Optional[str] = None):
    """Инициализирует глобальный экстрактор"""
    global _extractor
    logger = get_logger("XLM_RoBerta_entities_extractor")
    
    logger.info("Initializing global extractor")
    try:
        _extractor = XLM_RoBerta_entities_extractor(model_path, device)
        logger.info("Global extractor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize global extractor: {str(e)}")
        raise


def extract_entities(text: str) -> dict:
    """
    Извлекает сущности из текста.

    Args:
        text: Текст для анализа

    Returns:
        JSON-совместимый словарь с сущностями и их позициями
    """
    logger = get_logger("XLM_RoBerta_entities_extractor")
    
    if _extractor is None:
        logger.error("Global extractor not initialized. Call init_extractor() first")
        raise RuntimeError("Сначала вызовите init_extractor()")
    
    logger.debug("Extracting entities using global extractor")
    try:
        result = _extractor.extract_to_json(text)
        logger.debug("Entity extraction completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during entity extraction: {str(e)}")
        raise


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
    logger = get_logger("XLM_RoBerta_entities_extractor")
    
    if _extractor is None:
        logger.error("Global extractor not initialized. Call init_extractor() first")
        raise RuntimeError("Сначала вызовите init_extractor()")
    
    logger.debug(f"Anonymizing text with format: {placeholder_format}")
    try:
        result = _extractor.anonymize(text, placeholder_format, entity_types)
        logger.debug("Text anonymization completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error during text anonymization: {str(e)}")
        raise


# ============================================
# MAIN - демонстрация
# ============================================

if __name__ == "__main__":
    logger = get_logger("XLM_RoBerta_entities_extractor")
    logger.info("Starting XLM_RoBerta_entities_extractor demonstration")
    # Путь берётся из конфига автоматически
    try:
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
        
        logger.info("Demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error during demonstration: {str(e)}")
        raise