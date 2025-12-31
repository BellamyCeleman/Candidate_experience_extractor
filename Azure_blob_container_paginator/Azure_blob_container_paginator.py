from azure.storage.blob import BlobServiceClient
from .config import AzureBlobContainerConfig


class AzureBlobContainerPaginator:
    def __init__(self, config: AzureBlobContainerConfig):
        self.config = config

        self.blob_service = BlobServiceClient.from_connection_string(self.config.BLOB_CONN_STR)
        self.container_client = self.blob_service.get_container_client(
            self.config.BLOB_CONTAINER_NAME)

        self.blobs_iterator = self.container_client.list_blobs(
            results_per_page=self.config.page_size,
            name_starts_with=self.config.BLOB_PREFIX
        )

        self.pages = self.blobs_iterator.by_page()

    # def launch_cycle(self):
    #     count = 0
    #
    #     for page in self.pages:
    #         print("=== Новая страница ===")
    #
    #         for blob in page:
    #             count += 1
    #             print(f"Найден: {blob.name}")
    #
    #             if blob.name.endswith('.pdf'):
    #                 blob_client = self.container_client.get_blob_client(blob.name)
    #                 pdf_bytes = blob_client.download_blob().readall()
    #                 print(f"Скачан: {blob.name}, размер: {len(pdf_bytes)} байт")
    #
    #     print(f"Всего файлов: {count}")


if __name__ == "__main__":
    config = AzureBlobContainerConfig()
    print(f"Подключаюсь к контейнеру: {config.BLOB_CONTAINER_NAME}")

    paginator = AzureBlobContainerPaginator(config)