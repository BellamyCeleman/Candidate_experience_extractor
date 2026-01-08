# config.py
import os
from dotenv import load_dotenv

load_dotenv("dev.env")


def str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


class RFC_5424_LoggerConfig:
    def __init__(self):
        # Пути для логов
        self.LOCAL_LOG_FILE_PATH = os.getenv("OUTPUT_FILE_LOCAL_LOGFILE", "logs/app.log")
        self.APPLICATION_INSIGHTS_CONNECTION_STRING = os.getenv("APPLICATION_INSIGHTS_CONNECTION_STRING")

        # Флаги вывода
        self.LOG_TO_CONSOLE = str_to_bool(os.getenv("LOG_TO_CONSOLE", "true"))
        self.LOG_TO_FILE = str_to_bool(os.getenv("LOG_TO_FILE", "false"))
        self.LOG_TO_AZURE = str_to_bool(os.getenv("LOG_TO_AZURE", "false"))

        # Формат и уровень
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
        self.LOG_FORMAT = os.getenv(
            "LOG_FORMAT",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.LOG_DATE_FORMAT = os.getenv("LOG_DATE_FORMAT", "%Y-%m-%d %H:%M:%S")

    def get_log_level(self) -> int:
        """Конвертирует строковый уровень в константу logging."""
        import logging
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        return levels.get(self.LOG_LEVEL, logging.INFO)