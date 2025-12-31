import json
from openai import AzureOpenAI
from .config import ChatGPTConfig


class ChatGPT:
    """Client for Azure OpenAI API."""

    SYSTEM_PROMPT = """Ты эксперт по анализу резюме. Извлеки из текста следующие сущности и верни ТОЛЬКО валидный JSON без markdown-разметки:
{
    "full_name": "ФИО кандидата",
    "exp_dates": ["список дат с блока по опыту работы"],
    "positions": ["список должностей/профессий"],
    "locations": ["список городов/стран"],
    "skills": ["Список технических скиллов по типу: Azure, C++, Python..."],
    "companies": ["список компаний"],
    "education": ["список учебных заведений"]
}

Если какая-то информация отсутствует, оставь пустой список [] или null для full_name."""

    def __init__(self, model: str = "gpt-4o-mini"):
        config = ChatGPTConfig()
        self.model = model
        self.client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version="2024-12-01-preview",
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT
        )

    def extract_entities(self, resume_text: str) -> dict:
        """Extract structured entities from resume text."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": resume_text}
            ],
            max_tokens=2048,
            temperature=0.1
        )

        answer = response.choices[0].message.content
        return answer


if __name__ == "__main__":
    resume = """
    Образование:
    2014-2018 - КНУ им. Шевченко, Computer Science

    Иванов Иван Петрович
    Python Developer
    Киев, Украина

    Опыт работы:
    2020-2023 - Senior Developer в Google
    2018-2020 - Junior Developer в Startup Inc.

    Навыки: Python, Django, PostgreSQL, Docker, AWS
    """

    gpt = ChatGPT()
    result = gpt.extract_entities(resume)

    print(f"Имя: {result['full_name']}")
    print(f"Навыки: {result['skills']}")
    print(f"Должности: {result['positions']}")