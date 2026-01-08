import csv
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
   from Data_duration_extractor.ResumeDurationCalculator import ResumeDurationCalculator
   from Skills_extractor_from_text.SkillsExtractor import SkillsExtractor
except ImportError:
   from ResumeDurationCalculator import ResumeDurationCalculator
   from Skills_extractor_from_text.SkillsExtractor import SkillsExtractor


class ResumeAnalyzer:
   def __init__(self, skills_csv_path, output_file, model_name="phi4"):
      self.output_file = Path(output_file)
      self.skills_csv_path = Path(skills_csv_path)

      if not self.skills_csv_path.exists():
         raise FileNotFoundError(f"CRITICAL ERROR: Skills CSV not found at: {self.skills_csv_path}")

      self.duration_calc = ResumeDurationCalculator(model_name=model_name)
      self.skill_extractor = SkillsExtractor(path_to_csv=str(self.skills_csv_path))

   def analyze(self, text_content, date_string):
      print("--- Starting Analysis ---")

      print(f"Calculating duration for: '{date_string}'...")
      duration_result = self.duration_calc.get_duration(date_input=date_string)
      print(f"Duration found: {duration_result}")

      print("\nExtracting skills...")
      found_skills_set = self.skill_extractor.extract_skills(
         text=text_content,
         output_path=str(self.output_file)
      )
      found_skills_list = list(found_skills_set)
      print(f"Skills found: {found_skills_list}")

      years = duration_result.get('years', 0)
      months = duration_result.get('months', 0)

      with open(self.output_file, mode='w', newline='', encoding='utf-8') as file:
         writer = csv.writer(file)
         writer.writerow(["Skill", "Years", "Months"])

         for skill in found_skills_list:
            writer.writerow([skill, years, months])

      print(f"\n--- Result saved to CSV: {self.output_file} ---")
      return duration_result, found_skills_list


if __name__ == "__main__":
   current_script_dir = Path(__file__).resolve().parent
   project_root = current_script_dir.parent

   SKILLS_DB = project_root / "Skills_extractor_from_text" / "Skills_database" / "skills.csv"

   OUTPUT_CSV = project_root / "Data_duration_extractor" / "artifacts" / "results.csv"

   print(f"Looking for skills DB at: {SKILLS_DB}")

   resume_text = """
    Senior Backend Developer.
    Tech stack: Python, Django, PostgreSQL, Docker, Kubernetes.
    Experience with AWS and CI/CD pipelines.
    """

   resume_dates = "June 2021 - Present"

   analyzer = ResumeAnalyzer(
      skills_csv_path=SKILLS_DB,
      output_file=OUTPUT_CSV
   )

   analyzer.analyze(
      text_content=resume_text,
      date_string=resume_dates
   )