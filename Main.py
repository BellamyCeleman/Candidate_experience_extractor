"""
Точка входа в приложение.
Здесь настраивается логирование для всего проекта.
"""

# Init log system
from RFC_logging_system.LoggerFactory import configure_logging, get_logger

logger = get_logger("Main")

# Init paginator
from Azure_blob_container_paginator.Azure_blob_container_paginator import AzureBlobContainerPaginator
from Azure_blob_container_paginator.console_commands_for_paginator import ConsoleArgs


def main():

    args = (
        ConsoleArgs(description="Resume processing tool")
        .add("page-size", short="s", type=int, default=5, help="Resumes per page")
        .add("start-page", type=int, default=0, help="Start page")
        .add("end-page", type=int, default=3, help="End page")
        .add("test-files", short="t", flag=True, help="Enable PII verification")
        .add("token", help="Continuation token")
        .parse()
    )

    logger.info("Application started")

if __name__ == "__main__":
    main()