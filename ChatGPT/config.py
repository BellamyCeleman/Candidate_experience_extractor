import os
from dotenv import load_dotenv

load_dotenv("dev.env")

class ChatGPTConfig:
    def __init__(self):
        self.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        self.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
