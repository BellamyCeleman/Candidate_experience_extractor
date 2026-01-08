"""
Точка входа в приложение.
Здесь настраивается логирование для всего проекта.
"""

from RFC_logging_system.LoggerFactory import configure_logging, get_logger

# === НАСТРОЙКА ЛОГИРОВАНИЯ (ОДИН РАЗ) ===
# Все параметры меняются ТОЛЬКО ЗДЕСЬ
configure_logging(
    log_to_console=True,
    log_to_file=True,
    log_to_azure=False,
    log_file_path="logs/application.log",  # ← Меняешь путь здесь
    log_level="INFO",  # ← Меняешь уровень здесь
)

# Логгер для main модуля
logger = get_logger("Main")


def main():
    logger.info("Application started")

    # Пример использования ChatGPT_ErrorsCatcher
    from ChatGPT.ChatGPT_ErrorsCatcher import ChatGPT_EntitiesCatcher

    catcher = ChatGPT_EntitiesCatcher()

    text = """
    Иванов Иван Петрович
    Python Developer в Google
    2020-2023
    Skills: Python, Django, AWS
    """

    no_entities, explanation = catcher.check_entities(text)

    if no_entities:
        logger.info("Текст чистый, сущностей нет")
    else:
        logger.warning(f"Найдены сущности:\n{explanation}")

    logger.info("Application finished")


if __name__ == "__main__":
    main()