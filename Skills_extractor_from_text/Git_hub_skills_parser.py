import requests
import pandas as pd

def get_skills(url):
   skills = []
   try:
      response = requests.get(url)

      if response.status_code == 200:
         data = response.json()

         for elem in data:
            filename = elem['name']

            if filename.endswith(".svg"):
               skill = filename.replace(".svg", "")

               if skill.endswith("-Dark") or skill.endswith("-Light"):
                  skill = skill.replace("-Dark", "").replace("-Light", "")

               if skill not in skills:
                  skills.append(skill)
      else:
         print("Error")
   except Exception as e:
      print(e)

   return skills

if __name__=='__main__':
   url = 'https://api.github.com/repos/tandpfun/skill-icons/contents/icons'

   skills = get_skills(url)
   pd.DataFrame(skills, columns=["Skill"]).to_csv("Skills/skills.csv", index=False)
