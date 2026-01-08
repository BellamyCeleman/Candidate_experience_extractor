# RFC_5424_Logger.py
import logging
from typing import Optional

from opencensus.ext.azure.log_exporter import AzureLogHandler

from .config import RFC_5424_LoggerConfig


class RFC_5424_Logger:
    """
    Логгер с поддержкой вывода в консоль, файл и Azure Application Insights.
    """

    def __init__(
            self,
            config: RFC_5424_LoggerConfig,
            logger_name: str = "RFC_5424_Logger"
    ):
        self.config = config
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(config.get_log_level())

        # Предотвращаем дублирование хендлеров при повторной инициализации
        if self.logger.handlers:
            self.logger.handlers.clear()

        self._formatter = self._create_formatter()
        self._setup_handlers()

    def _create_formatter(self) -> logging.Formatter:
        return logging.Formatter(
            self.config.LOG_FORMAT,
            datefmt=self.config.LOG_DATE_FORMAT
        )

    def _setup_handlers(self) -> None:
        """Настраивает все хендлеры согласно конфигу."""
        if self.config.LOG_TO_CONSOLE:
            self._add_console_handler()

        if self.config.LOG_TO_FILE:
            self._add_file_handler()

        if self.config.LOG_TO_AZURE:
            self._add_azure_handler()

    def _add_console_handler(self) -> None:
        handler = logging.StreamHandler()
        handler.setLevel(self.logger.level)
        handler.setFormatter(self._formatter)
        self.logger.addHandler(handler)

    def _add_file_handler(self) -> None:
        handler = logging.FileHandler(
            self.config.LOCAL_LOG_FILE_PATH,
            encoding="utf-8"
        )
        handler.setLevel(self.logger.level)
        handler.setFormatter(self._formatter)
        self.logger.addHandler(handler)

    def _add_azure_handler(self) -> None:
        conn_str = self.config.APPLICATION_INSIGHTS_CONNECTION_STRING
        if not conn_str:
            self.logger.warning("Azure logging enabled but connection string is missing")
            return

        handler = AzureLogHandler(connection_string=conn_str)
        handler.setLevel(self.logger.level)
        handler.setFormatter(self._formatter)
        self.logger.addHandler(handler)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str, exc_info: bool = True) -> None:
        self.logger.error(message, exc_info=exc_info)

    def critical(self, message: str, exc_info: bool = True) -> None:
        self.logger.critical(message, exc_info=exc_info)

    def log(self, level: int, message: str) -> None:
        """Универсальный метод логирования с указанием уровня."""
        self.logger.log(level, message)


if __name__ == "__main__":
    config = RFC_5424_LoggerConfig()
    logger = RFC_5424_Logger(config, logger_name="TestLogger")

    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")