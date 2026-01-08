import os
from dotenv import load_dotenv

load_dotenv("dev.env")

class AzureBlobContainerConfig:
    def __init__(self):

        self.BLOB_CONN_STR = os.getenv("BLOB_CONN_STR")
        self.BLOB_CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")
        self.BLOB_PREFIX = os.getenv("BLOB_PREFIX")
        self.FULL_PATH_TO_BLOB = self.BLOB_PREFIX + self.BLOB_CONTAINER_NAME
        self.CURRENT_PAGE_TOKEN = None

        self.page_number = 0
        self.page_size = 0
        self.page_start = 0
        self.page_end = 0
        self.page_elements_counter = 0

