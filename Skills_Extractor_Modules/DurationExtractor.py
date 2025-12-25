import re
from datetime import datetime

from dateutil.relativedelta import relativedelta
from typing import Dict

from Project_for_using_llm.Skills_Extractor_Modules.SkillsExtractor import SkillsExtractor

extractor = SkillsExtractor("Project_for_using_llm/Skills_extractor_Modules/Skills/skills.csv")

class ExtractDuration:
   def __init__(self, path_to_skills: str) -> None:
      self.path_to_skills = path_to_skills

      self.extractor = SkillsExtractor(self.path_to_skills)

   def convert_to_datetime(self, date_str: str) -> datetime:
      date_str = date_str.strip()
      date_str = re.sub(r"\s+", " ", date_str)

      if date_str.lower() == "present":
         date = datetime.now()

      date = None

      formats = [
         "%m.%Y",
         "%m/%Y",
         "%m\\%Y",
         "%B %Y",
         "%b %Y",
         "%Y"
      ]
      for format in formats:
         try:
            date = datetime.strptime(date_str, format)
         except ValueError:
            continue

      if date is None:
         print(f"Warning: Could not parse date '{date_str}'")
         return datetime.now()

      if date.day > 14:
         if date.month == 12:
            return date.replace(year=date.year + 1, month=1, day=1)

         return datetime.replace(year=date.year, month=date.month + 1, day=1)

      return date.replace(day=1)

   def get_skills_duration(self, text: str) -> Dict[str, float]:
      present_skills = self.extractor.extract_skills(
         text,
         "Project_for_using_llm/Artifacts/extracted_skills/skills.json"
      )

      skills_duration = dict.fromkeys(present_skills, 0)

      pattern = r"([A-Za-z]+\s+\d{4}|\d{2}\.\d{4})\s*[-–—]\s*(PRESENT|[A-Za-z]+\s+\d{4}|\d{2}\.\d{4})"
      date_pattern = re.compile(pattern, re.IGNORECASE)

      matches = list(re.finditer(date_pattern, text))
      for i, match in enumerate(matches):
         start, end = match[1], match[2]

         start = self.convert_to_datetime(start)
         end = self.convert_to_datetime(end)

         difference = relativedelta(end, start)

         start_ind = match.end()
         end_ind = matches[i + 1].start() if len(matches) > i + 1 else len(text) - 1

         text_segment = text[start_ind:end_ind]

         segment_skills = self.extractor.extract_skills(text_segment, "Project_for_using_llm/Artifacts/extracted_skills.json")
         for skill in segment_skills:
            skills_duration[skill] += difference.years * 12
            skills_duration[skill] += difference.months

      return skills_duration

if __name__ == "__main__":
   resume_text = """
[100001_Person_7]
Senior PHP Developer
PERSON[100001_Person]NFORMATION
● Phone: +38[100001_PhoneNumber], [100001_SocialMedia]: [100001_Person_6]
● E­mail: [[100001_SocialMedia]
● [100001_SocialMedia] Profile: [[100001_SocialMedia]
● Age: 31 years old, Location: [100001_Location_6]
SK[100001_Person_3]S
Technical Skills
Programming Languages ● PHP (5.3+, up to 5.5)
○ Frameworks: Symfony 2, Laravel 4, Zend
Framework 1, Silex
○ Testing: PHPUnit, Behat, PhpSpec,
Codeception
○ Template engines: Twig, Smarty, Blade
● JavaScript, jQuery, WebSocket
● Python (basic knowledge)
Databases and data ● MySQL (complex queries, database design,
storages optimization, administration)
● MongoDB
● Redis, Memcached
Other Software and ● HTML5/XHTML
Technologies ● CSS
● Twitter Bootstrap
● [100001_SocialMedia], Gitlab, [100001_SocialMedia] ● WordPress (themes and plugins development,
optimization, security)
Operating Systems ● Debian/Ubuntu Linux
● Apache2, Nginx, Vagrant
IDEs and editors ● PHPStorm, Sublime Text, vim
Language Skills
● English (upper­intermediate)
● Ukrainian/Russian (fluent)
● Polish (pre­intermediate)
CV: [100001_Person_2]. Page 1 of 2
WORK EXPERIENCE
Python Developer
GloboTech
June 2013 – Present (1 year 5 months) | [100001_Location_5]
Technologies: 5.4, Silex, Symfony, MongoDB, BDD: Behat & PHPSpec, JavaScript.

Freelance PHP Developer
Self Employed
December 2008 – May 2013 (4 years 6 months)
Development and support of e­commerce websites and informational web portals.
Technologies: PHP 5.3, ZF1, WP, JQuery.
Frontend Developer
Delta Soft
May 2008 – November 2008 (7 months) | [100001_Location_4]
PSD to HTML coding. Creating v[100001_Person]d, well structured, semantic, seo­friendly html markup.
JavaScript development. Code qu[100001_Person]ty assurance.
Freelance WordPress Developer
Self Employed
October 2006 – April 2008 (1 year 7 months)
WordPress themes and plugins development, security, optimization.
SEO/SEM Speci[100001_Person]st
Internet Systems
March 2006 – September 20[100001_PhoneNumber]months) | [100001_Location_3]
Search engine optimization, search engine marketing, development of SEO strategies,
development of SEO recommendations for technical speci[100001_Person]sts, web analytics.
PHP Developer
Part­time Self Employed
2004 – 2006 (2 years)
Search engine optimization tools development. Multi­process programming using pcntl and curl
extensions, performance optimization, proxying requests.
EDUCATION
Master of Ecological Engineering ­ [100001_Location_2]"
December 2000 – February 2006
"Introduction to Databases" ­ free online class by [100001_Location_1].
October 2011 – December 2011
Grade: 97%. [100001_SocialMedia]
CV: [100001_Person_1]. Page 2 of 2
   """

   extract_duration = ExtractDuration("Project_for_using_llm/Skills_extractor_Modules/Skills/skills.csv")

   print(extract_duration.get_skills_duration(resume_text))