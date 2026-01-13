import ollama
import json
import datetime
import re


class ResumeDurationCalculator:
   def __init__(self, model_name: str = "phi4", *, auto_pull_model: bool = True):
      self.model_name = model_name

      # Ensure required Ollama model is available locally (e.g. "phi4").
      # Keep imports resilient whether this file is run as a module or directly.
      try:
         from Data_duration_extractor.ollama_model_manager import ensure_ollama_model
      except Exception:
         from ollama_model_manager import ensure_ollama_model  # type: ignore

      ensure_ollama_model(self.model_name, pull_if_missing=auto_pull_model)

   def get_duration(self, date_input: str) -> dict:
      current_date_str = datetime.date.today().strftime("%Y-%m-%d")

      prompt = f"""
         Role: Resume Duration Calculator
         Current Reference Date: {current_date_str}
         
         Task: Extract start and end dates from the input, calculate the duration, and output the result in JSON. Do not write explanations of your answer!
         
         Examples:
         Input: August 2020 - August 2021
         Output: {{"years": 1, "months": 0}}
         
         Input: Jan 2023 - Present
         Output: {{"years": 2, "months": 11}} (Calculated based on Current Reference Date)
         
         Input: 01/2020 - 03/2020
         Output: {{"years": 0, "months": 2}}
         
         Input: {date_input}
         Output:
         """

      try:
         response = ollama.chat(model=self.model_name, messages=[
            {
               'role': 'user',
               'content': prompt,
            },
         ])

         content = response['message']['content']

         clean_content = re.sub(r'```json\s*|\s*```', '', content).strip()

         result_json = json.loads(clean_content)

         return result_json

      except Exception as e:
         print(f"An error occurred: {e}")
         return {"years": 0, "months": 0}

if __name__ == "__main__":
   calculator = ResumeDurationCalculator(model_name="phi4")

   test_dates = [
      "March 2018 - April 2020",
      "01/2022 - Present",
      "2015 - 2016"
   ]

   for dates in test_dates:
      result = calculator.get_duration(dates)
      print(f"Input: '{dates}'")
      print(f"Output: {result}")
      print("-" * 30)