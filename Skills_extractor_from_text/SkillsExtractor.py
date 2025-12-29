"""
Module for extracting skills from text.

Uses a Trie (prefix tree) data structure for efficient skill matching
against a predefined dictionary of skills.

Usage example:
    extractor = SkillsExtractor("path/to/skills.csv")
    found = extractor.extract_skills("I know Python and SQL", "output.json")
"""

from pathlib import Path
import json
import csv
from typing import List, Dict, Set


class SkillsExtractor:
    """
    Extracts skills from text using a Trie-based search algorithm.

    The Trie structure allows finding all matching skills in a single
    pass through the text, which is more efficient than checking
    each skill individually (O(n*m) vs O(n*k) where n=text length,
    m=longest skill length, k=number of skills).

    Attributes:
        path_to_csv: Path to CSV file containing the skills dictionary.
        tree: Trie structure for fast skill lookup.
    """

    def __init__(self, path_to_csv: str):
        """
        Initialize the extractor and build the Trie from skills list.

        Args:
            path_to_csv: Path to CSV file where the first column
                         contains possible skill names.
        """
        self.path_to_csv = path_to_csv
        self.tree = self._create_search_dict()

    def _create_search_dict(self) -> Dict:
        """
        Build a Trie (prefix tree) from the skills list.

        Each skill is added character by character. The '$' key marks
        the end of a valid skill and stores its original form.

        Example structure for skills ["Python", "SQL"]:
        {
            'p': {'y': {'t': {'h': {'o': {'n': {'$': 'Python'}}}}}},
            's': {'q': {'l': {'$': 'SQL'}}}
        }

        Returns:
            Dict representing the Trie structure.
        """
        skills = self.get_possible_skills()
        tree = {}

        for skill in skills:
            skill_normalized = skill.lower()
            current = tree

            # Traverse/build the tree character by character
            for char in skill_normalized:
                if char not in current:
                    current[char] = {}
                current = current[char]

            # Mark end of skill, store original case
            current['$'] = skill

        return tree

    def get_possible_skills(self) -> List[str]:
        """
        Load skills from the CSV file.

        Reads the first column of each row as a skill name.

        Returns:
            List of skill names from the CSV file.
        """
        skills = []

        with open(self.path_to_csv, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:
                    skills.append(row[0])
        return skills

    @staticmethod
    def _check_word_ended(text: str, start: int, end: int) -> bool:
        """
        Check if the substring is a complete word (not part of another word).

        Ensures the match is bounded by non-alphanumeric characters
        (spaces, punctuation, or string boundaries).

        Args:
            text: The full text being searched.
            start: Start index of the potential match.
            end: End index of the potential match.

        Returns:
            True if the substring is a standalone word, False otherwise.

        Example:
            "Python" in "I love Python!" -> True
            "SQL" in "MySQL" -> False (part of larger word)
        """
        # Check if there's an alphanumeric character before the match
        if start > 0 and text[start - 1].isalnum():
            return False
        # Check if there's an alphanumeric character after the match
        elif end < len(text) - 1 and text[end + 1].isalnum():
            return False

        return True

    def extract_skills(self, text: str, output_path: str) -> Set[str]:
        """
        Extract all matching skills from the given text.

        Performs a single pass through the text, checking at each position
        if any skill from the Trie matches starting from that position.

        Args:
            text: Input text to search for skills.
            output_path: Path where found skills will be saved as JSON.

        Returns:
            Set of found skill names (with original casing from CSV).

        Example:
            extractor.extract_skills("Senior Python Developer", "out.json")
            {'Python'}
        """
        output_path = Path(output_path)
        found_skills = set()

        # Normalize text: collapse whitespace and convert to lowercase
        text_normalized = " ".join(text.split()).lower()
        length = len(text_normalized)

        # Try to match a skill starting at each position
        for start in range(length):
            current = self.tree
            end = start

            # Follow the Trie as long as characters match
            while end < length and text_normalized[end] in current:
                current = current[text_normalized[end]]

                # Check if we've reached the end of a valid skill
                if '$' in current:
                    if self._check_word_ended(text_normalized, start, end):
                        found_skills.add(current['$'])  # Add original cased skill
                end += 1

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results as JSON
        with open(output_path, 'w') as f:
            json.dump(list(found_skills), f)

        return found_skills


if __name__ == "__main__":
    # Demo: Extract skills from sample text

    # 1. Create extractor with path to skills dictionary
    extractor = SkillsExtractor("Project_for_using_llm/Skills_extractor_Modules/Skills/skills.csv")

    # 2. Sample text to analyze
    text = """
    Senior Developer with 5 years of experience.
    Skills: Python, JavaScript, Docker, Kubernetes, 
    machine learning, PostgreSQL, FastAPI, Git.
    """

    # 3. Extract skills and save to file
    found = extractor.extract_skills(
        text,
        "Project_for_using_llm/Artifacts/extracted_skills/skills.json"
    )

    print(f"Found skills: {found}")