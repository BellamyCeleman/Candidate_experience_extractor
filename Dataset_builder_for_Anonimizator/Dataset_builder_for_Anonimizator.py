# For intialization console arguments
from Azure_blob_container_paginator.console_commands_for_paginator import ConsoleArgs

# Paginator by blob files
from Azure_blob_container_paginator.Azure_blob_container_paginator import AzureBlobContainerPaginator
from Azure_blob_container_paginator.config import AzureBlobContainerConfig

# For extraction entities
from ChatGPT.ChatGPT import ChatGPT

# For converting pdf-files to text
from File_convertors.PDF_to_TXT_converter import PDFToTextConverter

# For saving text to custom path
from File_manager.File_manager import FileManager

def launch_paginator_cycle(paginator_config, paginator, pdf_converter, entities_extractor, file_manager):

    count = paginator_config.page_elements_counter

    if paginator_config.page_start > paginator_config.page_end:
        return

    for page in paginator.pages:

        print("-"*20, f"PAGE NUMBER: {paginator_config.page_number}", "-"*20, "\n")

        if paginator_config.page_number > paginator_config.page_end:
            paginator_config.page_number += 1

        for blob in page:
            print(f"Найден: {blob.name}")

            if blob.name.endswith('.pdf'):

                blob_client = paginator.container_client.get_blob_client(blob.name)

                pdf_bytes = blob_client.download_blob().readall()

                extracted_text = pdf_converter.convert(pdf_bytes)

                extracted_entities = entities_extractor.extract_entities(extracted_text)

                print(extracted_entities)

                file_manager.save("output.txt", extracted_entities, append="a")
                file_manager.save("output.txt", "\n", append="a")

            paginator_config.page_elements_counter += 1

    print(f"Всего файлов: {count}")

if __name__ == "__main__":
    # Init console flags
    args = (
        ConsoleArgs(description="Resume processing tool")
        .add("page-size", short="s", type=int, default=None, help="Resumes per page")
        .add("start-page", type=int, default=None, help="Start page")
        .add("end-page", type=int, default=None, help="End page")
        .add("test-files", short="t", type=bool, default=False, flag=True, help="Enable PII verification")
        .add("token", help="Continuation token")
        .parse()
    )

    # PDF to text converter
    text_converter = PDFToTextConverter()

    # Chat gpt for extracting entities
    entities_extractor = ChatGPT()

    # File manager for saving text to custom file
    file_manager = FileManager(base_dir="datasets")

    # Init paginator args
    paginator_config = AzureBlobContainerConfig()
    paginator_config.page_size = args.page_size
    paginator_config.page_start = args.start_page
    paginator_config.page_end = args.end_page
    paginator_config.test_files = args.test_files
    paginator_config.token = args.token

    # Init paginator
    paginator = AzureBlobContainerPaginator(paginator_config)
    launch_paginator_cycle(paginator_config,paginator=paginator, pdf_converter=text_converter, entities_extractor=entities_extractor,
                           file_manager=file_manager)




