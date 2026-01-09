import os
from dotenv import load_dotenv
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging

# --- Helpers ---
def str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")

# --- Load .env ---
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "dev.env"))

OUTPUT_FILE_LOCAL_LOGFILE = os.getenv("OUTPUT_FILE_LOCAL_LOGFILE")
BLOB_CONN_STR = os.getenv("BLOB_CONN_STR")
OUTPUT_CONTAINER_AZURE_FOR_LOGS = os.getenv("OUTPUT_CONTAINER_AZURE_FOR_LOGS", "logs")
OUTPUT_BLOB_FILE_FOR_LOGS = os.getenv("OUTPUT_BLOB_FILE_FOR_LOGS", "application.log")
APPLICATION_INSIGHTS_CONNECTION_STRING = os.getenv("APPLICATION_INSIGHTS_CONNECTION_STRING")

LOG_TO_CONSOLE = str_to_bool(os.getenv("LOG_TO_CONSOLE", "false"))
LOG_TO_FILE = str_to_bool(os.getenv("LOG_TO_FILE", "false"))
LOG_TO_AZURE = str_to_bool(os.getenv("LOG_TO_AZURE", "false"))


class RFC_5424_Logger:
    """File"""

    def __init__(self, local_log_file_path=OUTPUT_FILE_LOCAL_LOGFILE, massage_level=logging.INFO, logger_name="RFC_5424_Logger",
                 log_to_file=LOG_TO_FILE, log_to_azure=LOG_TO_AZURE, log_to_console=LOG_TO_CONSOLE):

        self.local_log_file_path = local_log_file_path
        self.message_level = massage_level
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(massage_level)

        self.log_to_file = log_to_file
        self.log_to_azure = log_to_azure
        self.log_to_console = log_to_console

        # Basic format
        formatter = logging.Formatter(
            f'%(asctime)s - {self.logger_name} - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        #  Initialize FileHandler and StreamHandler
        self.file_handler = logging.FileHandler("Project_for_using_llm/Artifacts/logs/general_log.log")
        self.console_handler = logging.StreamHandler()

        self.file_handler.setLevel(massage_level)
        self.file_handler.setFormatter(formatter)

        self.console_handler.setLevel(massage_level)
        self.console_handler.setFormatter(formatter)

        # Azure handler
        self.azure_handler = None
        if APPLICATION_INSIGHTS_CONNECTION_STRING:
            self.azure_handler = AzureLogHandler(connection_string=APPLICATION_INSIGHTS_CONNECTION_STRING)
            self.azure_handler.setLevel(massage_level)
            self.azure_handler.setFormatter(formatter)

        # Attach handlers to class logger
        if LOG_TO_CONSOLE:
            self.logger.addHandler(self.console_handler)
        if LOG_TO_FILE:
            self.logger.addHandler(self.file_handler)
        if LOG_TO_AZURE and self.azure_handler:
            self.logger.addHandler(self.azure_handler)

    def send_message(self, logger_name, message, level_log=logging.INFO, send_to_azure=True):
        """
        Args:
            level_log: logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            logger_name: name of the logger
            message: message text
            send_to_azure: bool or None
                - True: send to Azure
                - False: do not send to в Azure
                - None: use value LOG_TO_AZURE
        """
        # For changing logger name
        self.change_log_name(logger_name)

        should_send_to_azure = self.log_to_azure if send_to_azure is None else send_to_azure

        if self.azure_handler:
            if should_send_to_azure and self.azure_handler not in self.logger.handlers:
                self.logger.addHandler(self.azure_handler)
            elif not should_send_to_azure and self.azure_handler in self.logger.handlers:
                self.logger.removeHandler(self.azure_handler)

        # Log message
        if level_log == logging.DEBUG:
            self.logger.debug(message)
        elif level_log == logging.INFO:
            self.logger.info(message)
        elif level_log == logging.WARNING:
            self.logger.warning(message)
        elif level_log == logging.ERROR:
            self.logger.error(message, exc_info=True)  # ← Добавил exc_info=True
        elif level_log == logging.CRITICAL:
            self.logger.critical(message)

    def change_log_name(self, logger_name):
        self.logger_name = logger_name

        # For changing logger name in format
        formatter = logging.Formatter(
            f'%(asctime)s - {self.logger_name} - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        for handler in self.logger.handlers:
            handler.setFormatter(formatter)


if __name__ == "__main__":
    logger = RFC_5424_Logger(local_log_file_path=OUTPUT_FILE_LOCAL_LOGFILE)

    # Пример 1: используется глобальное значение LOG_TO_AZURE
    logger.send_message(level_log=logging.ERROR, logger_name="Another_name", message="Something", send_to_azure=True)

    # Пример 2: явно отправить в Azure
    logger.send_message(level_log=logging.INFO, logger_name="Critical", message="Important event", send_to_azure=True)

    # Пример 3: НЕ отправлять в Azure
    logger.send_message(level_log=logging.DEBUG, logger_name="Debug", message="Debug info", send_to_azure=True)