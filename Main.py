"""
Resume Anonymization Pipeline
Скачивает резюме из Azure Blob → Анонимизирует → Проверяет на PII
"""

from RFC_logging_system.LoggerFactory import configure_logging, get_logger
from Azure_blob_container_paginator.Azure_blob_container_paginator import AzureBlobContainerPaginator
from Azure_blob_container_paginator.config import AzureBlobContainerConfig
from Azure_blob_container_paginator.console_commands_for_paginator import ConsoleArgs
from XLM_RoBerta_entities_extractor.XLM_RoBerta_entities_extractor import init_extractor, anonymize_text
from ChatGPT.ChatGPT_EntitiesCatcher import ChatGPT_EntitiesCatcher
from File_convertors.PDF_to_TXT_converter import PDFToTextConverter
import os

logger = get_logger("Main")


# ─────────────────────────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────────────────────────

def extract_text(blob_client) -> str:
    """Извлекает текст из PDF или TXT файла."""
    name = blob_client.blob_name
    data = blob_client.download_blob().readall()

    if name.lower().endswith('.pdf'):
        return PDFToTextConverter().convert(data) or ""

    if name.lower().endswith('.txt'):
        return data.decode('utf-8', errors='ignore')

    logger.warning(f"Unsupported format: {name}")
    return ""


def process_resume(text: str, name: str, verify_with_gpt: bool) -> bool:
    """
    Анонимизирует резюме и проверяет на остатки PII.

    Returns:
        True = успех (PII не найдено), False = провал (PII найдено или ошибка)
    """

    # 1. Анонимизация
    try:
        result = anonymize_text(text, placeholder_format="[REDACTED]")
        anon_text = result["anonymized_text"]
        replacements = result["replacements"]
        logger.info(f"{name}: {len(replacements)} entities replaced")
    except Exception as e:
        logger.error(f"{name}: Anonymization failed - {e}")
        return False

    # 2. Проверка GPT (если включена)
    if not verify_with_gpt:
        return True

    try:
        catcher = ChatGPT_EntitiesCatcher()
        is_clean, explanation = catcher.check_entities(anon_text)

        if not is_clean:
            logger.error(f"{name}: FAILED - PII still found: {explanation}")
            return False

        logger.info(f"{name}: PASSED - no PII detected")
        return True

    except Exception as e:
        logger.error(f"{name}: GPT check failed - {e}")
        return False


def process_batch(paginator: AzureBlobContainerPaginator,
                  start: int, end: int, verify: bool) -> tuple[int, int, int]:
    """
    Обрабатывает страницы резюме из Azure Blob.

    Returns:
        (total, passed, failed) — счётчики
    """
    total = passed = failed = 0
    pages = paginator.blobs_iterator.by_page()

    for page_num in range(end + 1):
        try:
            page = next(pages)
        except StopIteration:
            logger.info(f"No more pages after {page_num - 1}")
            break

        if page_num < start:
            logger.debug(f"Skipping page {page_num}")
            continue

        logger.info(f"─── Page {page_num} ───")

        for blob in page:
            total += 1

            blob_client = paginator.container_client.get_blob_client(blob.name)
            text = extract_text(blob_client)

            if not text.strip():
                logger.warning(f"{blob.name}: Empty file")
                failed += 1
                continue

            if process_resume(text, blob.name, verify):
                passed += 1
            else:
                failed += 1

    return total, passed, failed


def export_for_labeling(paginator: AzureBlobContainerPaginator, start: int, end: int):
    """
    Скачивает резюме, разбивает на слова и сохраняет в файл.
    Каждое слово с новой строки. Между резюме — пустая строка.
    """

    # Путь к файлу
    output_dir = "Candidate_experience_extractor"
    output_file = os.path.join(output_dir, "dataset_labeling.txt")

    # Создаем папку, если её нет
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Writing dataset to: {output_file}")

    # Открываем файл в режиме 'a' (append), чтобы дописывать данные, а не перезатирать
    # Если нужно перезатирать каждый раз, замените 'a' на 'w'
    with open(output_file, "a", encoding="utf-8") as f:

        pages = paginator.blobs_iterator.by_page()
        processed_count = 0

        for page_num in range(end + 1):
            try:
                page = next(pages)
            except StopIteration:
                break

            if page_num < start:
                continue

            logger.info(f"Processing Page {page_num}...")

            for blob in page:
                blob_client = paginator.container_client.get_blob_client(blob.name)
                text = extract_text(blob_client)

                if not text.strip():
                    continue

                # 1. Разбиваем текст на слова (split по умолчанию делит по пробелам, табам и энтерам)
                words = text.split()

                if not words:
                    continue

                # 2. Записываем слова: каждое с новой строки
                f.write('\n'.join(words))

                # 3. Добавляем разделитель (пустую строку) между резюме
                # Одиночный \n перенесет курсор после последнего слова
                # Двойной \n создаст пустую строку между блоками
                f.write('\n\n')

                processed_count += 1
                logger.info(f"Saved: {blob.name} ({len(words)} words)")

    logger.info(f"Done. Processed {processed_count} resumes.")

# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

def main():

    logger.info("Starting Resume Anonymization Pipeline")

    # CLI args
    args = (
        ConsoleArgs(description="Resume anonymization tool")
        .add("page-size", short="s", type=int, default=5, help="Resumes per page")
        .add("start-page", type=int, default=0, help="Start page")
        .add("end-page", type=int, default=3, help="End page")
        .add("test-files", short="t", flag=True, help="Verify with ChatGPT")
        .parse()
    )

    # Init
    config = AzureBlobContainerConfig()
    config.page_size = args.page_size

    paginator = AzureBlobContainerPaginator(config)
    paginator.blobs_iterator = paginator.container_client.list_blobs(
        results_per_page=args.page_size,
        name_starts_with=config.BLOB_PREFIX
    )

    export_for_labeling(paginator, args.start_page, args.end_page)

    # init_extractor()
    # logger.info("Components initialized")
    #
    # # Process
    # total, passed, failed = process_batch(
    #     paginator, args.start_page, args.end_page, args.test_files
    # )
    #
    # # Summary
    # logger.info("=" * 50)
    # logger.info(f"TOTAL:  {total}")
    # logger.info(f"PASSED: {passed}")
    # logger.info(f"FAILED: {failed}")
    # logger.info(f"Rate:   {passed / max(total, 1) * 100:.1f}%")
    # logger.info("=" * 50)


if __name__ == "__main__":
    main()