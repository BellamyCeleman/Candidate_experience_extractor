"""
Фабрика логгеров - единая точка доступа ко всем логгерам проекта.

"""

from typing import Optional, Dict
from threading import Lock

from .config import RFC_5424_LoggerConfig
from .RFC_5424_Logger import RFC_5424_Logger


class LoggerFactory:
    """
    Singleton-фабрика для создания и управления логгерами.

    Гарантирует:
    - Единую конфигурацию для всех логгеров
    - Один экземпляр логгера на имя (кэширование)
    - Потокобезопасность
    """

    _instance: Optional["LoggerFactory"] = None
    _lock: Lock = Lock()

    def __new__(cls) -> "LoggerFactory":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._config: Optional[RFC_5424_LoggerConfig] = None
        self._loggers: Dict[str, RFC_5424_Logger] = {}
        self._loggers_lock = Lock()
        self._initialized = True

    def configure(
            self,
            log_to_console: bool = True,
            log_to_file: bool = False,
            log_to_azure: bool = False,
            log_file_path: str = "logs/application.log",
            log_level: str = "INFO",
            log_format: Optional[str] = None,
            azure_connection_string: Optional[str] = None
    ) -> "LoggerFactory":
        """
        Настраивает глобальную конфигурацию логгера.

        Вызывать ОДИН РАЗ при старте приложения (в main.py или __init__.py).

        Args:
            log_to_console: Выводить в консоль
            log_to_file: Записывать в файл
            log_to_azure: Отправлять в Azure Application Insights
            log_file_path: Путь к файлу логов
            log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Кастомный формат (опционально)
            azure_connection_string: Строка подключения к Azure (опционально)

        Returns:
            self для цепочки вызовов
        """
        self._config = RFC_5424_LoggerConfig()

        # Переопределяем настройки
        self._config.LOG_TO_CONSOLE = log_to_console
        self._config.LOG_TO_FILE = log_to_file
        self._config.LOG_TO_AZURE = log_to_azure
        self._config.LOCAL_LOG_FILE_PATH = log_file_path
        self._config.LOG_LEVEL = log_level.upper()

        if log_format:
            self._config.LOG_FORMAT = log_format

        if azure_connection_string:
            self._config.APPLICATION_INSIGHTS_CONNECTION_STRING = azure_connection_string

        # Пересоздаём все существующие логгеры с новой конфигурацией
        with self._loggers_lock:
            for name in list(self._loggers.keys()):
                self._loggers[name] = RFC_5424_Logger(self._config, logger_name=name)

        return self

    def get_logger(self, name: str) -> RFC_5424_Logger:
        """
        Получает или создаёт логгер с указанным именем.

        Args:
            name: Имя логгера (обычно имя класса или модуля)

        Returns:
            Экземпляр RFC_5424_Logger
        """
        with self._loggers_lock:
            if name not in self._loggers:
                config = self._config or RFC_5424_LoggerConfig()
                self._loggers[name] = RFC_5424_Logger(config, logger_name=name)
            return self._loggers[name]

    def get_config(self) -> RFC_5424_LoggerConfig:
        """Возвращает текущую конфигурацию."""
        if self._config is None:
            self._config = RFC_5424_LoggerConfig()
        return self._config

    def reset(self) -> None:
        """Сбрасывает фабрику (для тестов)."""
        with self._loggers_lock:
            self._loggers.clear()
            self._config = None


# === Удобные функции для импорта ===

_factory = LoggerFactory()


def configure_logging(**kwargs) -> LoggerFactory:
    """
    Настраивает логирование для всего проекта.

    Пример:
        configure_logging(
            log_to_console=True,
            log_to_file=True,
            log_file_path="logs/app.log",
            log_level="DEBUG"
        )
    """
    return _factory.configure(**kwargs)


def get_logger(name: str) -> RFC_5424_Logger:
    """
    Получает логгер по имени.

    Пример:
        logger = get_logger("ChatGPT_EntitiesCatcher")
        logger.info("Starting entity extraction...")
    """
    return _factory.get_logger(name)


def get_config() -> RFC_5424_LoggerConfig:
    """Возвращает текущую конфигурацию логгера."""
    return _factory.get_config()


if __name__ == "__main__":
    # Пример использования

    # 1. Настраиваем один раз при старте
    configure_logging(
        log_to_console=True,
        log_to_file=True,
        log_file_path="logs/test.log",
        log_level="DEBUG"
    )

    # 2. Используем в любом месте проекта
    logger1 = get_logger("Module1")
    logger2 = get_logger("Module2")

    logger1.info("Message from Module1")
    logger2.warning("Warning from Module2")

    # 3. Тот же логгер возвращается при повторном вызове
    logger1_again = get_logger("Module1")
    assert logger1 is logger1_again  # True