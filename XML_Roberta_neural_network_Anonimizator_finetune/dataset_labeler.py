"""
Dataset Entity Labeler using ChatGPT API
Converts raw resume text to CoNLL format with entity labels
"""

import json
import re
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple
from openai import AzureOpenAI, BadRequestError, RateLimitError
from dataclasses import dataclass
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ChatGPT.config import ChatGPTConfig


@dataclass
class Entity:
    """Entity with type, text, and character positions"""
    type: str
    text: str
    start: int
    end: int


class DatasetLabeler:
    """Labels entities in resume text using ChatGPT API"""
    
    SYSTEM_PROMPT = """You are an expert in Named Entity Recognition (NER) for resume analysis.

Your task is to analyze resume text and extract ALL occurrences of the following entities:

1. DATES - Any dates, periods, years (e.g., "2020-2023", "January 2021", "2019", "OCT 2021", "DEC 2022", "15.02.1982")
2. COMPANIES - Names of companies, organizations where a person worked (e.g., "Google", "Microsoft", "BitAlpha")
3. HARD_SKILLS - Technical skills ONLY (programming languages, frameworks, tools, technologies like "Python", "SQL", "Docker", "AWS", "Tableau"). Do NOT include soft skills.
4. FULL_NAME - Person's full name (first name, last name, patronymic if present)
5. LOCATIONS - Geographic locations: cities, countries, addresses, regions (e.g., "Ukraine", "Kyiv", "Vienna", "New York", "Lviv", "Kyiv/Vienna")

IMPORTANT RULES:
- Extract EVERY occurrence of each entity type
- Provide the EXACT text as it appears in the resume
- List each occurrence separately (even if the same entity appears multiple times)
- Return ONLY valid JSON, no markdown formatting
- Be thorough - don't miss any entities
- Do NOT include soft skills like "communication", "teamwork", "leadership"
- Include all geographic locations: countries, cities, addresses, regions

Return JSON in this exact format:
{
    "FULL_NAME": ["John Smith", "Maria Garcia"],
    "DATES": ["2020-2023", "January 2021", "2019"],
    "COMPANIES": ["Google", "Microsoft"],
    "HARD_SKILLS": ["Python", "SQL", "Docker", "AWS"],
    "LOCATIONS": ["Ukraine", "Kyiv", "Vienna", "New York"]
}"""

    def __init__(self, model: str = "gpt-4o-mini"):
        config = ChatGPTConfig()
        self.model = model
        self.client = AzureOpenAI(
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version="2024-12-01-preview",
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT
        )
        
    def extract_entities(self, text: str, max_retries: int = 3) -> List[Entity]:
        """
        Extract entities from text using ChatGPT API
        
        Args:
            text: Resume text to analyze
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of Entity objects with type, text, and positions
        """
        if not text or not text.strip():
            return []
        
        user_message = f"Extract all entities from this resume text:\n\n{text}"
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=4096,
                    temperature=0.1
                )
                
                answer = response.choices[0].message.content
                
                # Parse JSON response
                cleaned_answer = answer.strip()
                if cleaned_answer.startswith("```"):
                    cleaned_answer = re.sub(r'^```(?:json)?\n?', '', cleaned_answer)
                    cleaned_answer = re.sub(r'\n?```$', '', cleaned_answer)
                
                data = json.loads(cleaned_answer)
                
                # Convert to Entity objects and find positions in text
                entities = []
                
                for entity_type, entity_texts in data.items():
                    if not isinstance(entity_texts, list):
                        continue
                    
                    for entity_text in entity_texts:
                        # Find all occurrences of this entity in the text
                        found_positions = self._find_entity_positions(text, entity_text)
                        
                        for start, end in found_positions:
                            entities.append(Entity(
                                type=entity_type,
                                text=entity_text,
                                start=start,
                                end=end
                            ))
                
                return entities
                
            except RateLimitError:
                wait_time = (attempt + 1) * 10
                print(f"Rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                
            except json.JSONDecodeError as e:
                print(f"JSON parse error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print(f"Raw response: {answer[:500]}")
                    return []
                time.sleep(2)
                
            except BadRequestError as e:
                print(f"Bad request error: {e}")
                return []
                
            except Exception as e:
                print(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return []
                time.sleep(2)
        
        return []
    
    def _find_entity_positions(self, text: str, entity_text: str) -> List[Tuple[int, int]]:
        """
        Find all occurrences of entity_text in text and return their positions
        
        Args:
            text: The full text to search in
            entity_text: The entity text to find
            
        Returns:
            List of (start, end) position tuples
        """
        positions = []
        search_text = entity_text.strip()
        
        if not search_text:
            return positions
        
        # Find all occurrences
        start_pos = 0
        while True:
            pos = text.find(search_text, start_pos)
            if pos == -1:
                break
            positions.append((pos, pos + len(search_text)))
            start_pos = pos + 1  # Move past this occurrence to find next
        
        return positions
    
    def entities_to_conll(self, text: str, entities: List[Entity]) -> List[Tuple[str, str]]:
        """
        Convert text and entities to CoNLL format (token, tag)
        Uses BIO tagging: B- (Begin), I- (Inside), O (Outside)
        
        Args:
            text: Original text
            entities: List of entities with positions
            
        Returns:
            List of (token, tag) tuples
        """
        # Tokenize text (simple whitespace tokenization)
        tokens = []
        
        # Split by whitespace but keep track of positions
        for match in re.finditer(r'\S+', text):
            token = match.group()
            start = match.start()
            end = match.end()
            tokens.append({
                'text': token,
                'start': start,
                'end': end,
                'tag': 'O'
            })
        
        # Map entity types to tag prefixes
        entity_type_map = {
            'FULL_NAME': 'PER',
            'DATES': 'DATE',
            'COMPANIES': 'ORG',
            'HARD_SKILLS': 'SKILL',
            'LOCATIONS': 'LOC'
        }
        
        # Sort entities by start position to process in order
        sorted_entities = sorted(entities, key=lambda e: e.start)
        
        # Assign tags based on entity positions
        for entity in sorted_entities:
            tag_prefix = entity_type_map.get(entity.type, 'O')
            if tag_prefix == 'O':
                continue
            
            # Find tokens that belong to this entity
            entity_tokens = []
            
            for i, token in enumerate(tokens):
                token_start = token['start']
                token_end = token['end']
                
                # Calculate overlap
                overlap_start = max(token_start, entity.start)
                overlap_end = min(token_end, entity.end)
                overlap_length = max(0, overlap_end - overlap_start)
                
                # Token belongs to entity if:
                # 1. Token is completely inside entity, OR
                # 2. Significant overlap (>50% of token length)
                token_length = token_end - token_start
                
                if overlap_length > 0:
                    # Check if majority of token overlaps with entity
                    if (token_start >= entity.start and token_end <= entity.end) or \
                       (overlap_length >= token_length * 0.5):
                        entity_tokens.append(i)
            
            # Tag the tokens
            for idx, token_idx in enumerate(entity_tokens):
                if idx == 0:
                    tokens[token_idx]['tag'] = f'B-{tag_prefix}'
                else:
                    tokens[token_idx]['tag'] = f'I-{tag_prefix}'
        
        return [(token['text'], token['tag']) for token in tokens]
    
    def format_conll_output(self, token_tag_pairs: List[Tuple[str, str]]) -> str:
        """Format token-tag pairs as CoNLL format string"""
        lines = [f"{token} {tag}" for token, tag in token_tag_pairs]
        return '\n'.join(lines)


def load_resumes(file_path: str) -> List[str]:
    """Load resumes from file (split by double newlines)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    resumes = [r.strip() for r in text.split('\n\n') if r.strip()]
    return resumes


def save_progress(output_path: str, labeled_data: List[str]):
    """Save labeled data to output file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(labeled_data))


def load_checkpoint(checkpoint_path: str) -> int:
    """Load the last processed resume index from checkpoint"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return int(f.read().strip())
    return 0


def save_checkpoint(checkpoint_path: str, index: int):
    """Save current progress to checkpoint"""
    with open(checkpoint_path, 'w') as f:
        f.write(str(index))


def main():
    """Main labeling pipeline"""
    
    # Configuration
    INPUT_FILE = "datasets/dataset_labeling.txt"
    OUTPUT_FILE = "XML_Roberta_neural_network_Anonimizator_finetune/dataset_labeled.conll"
    CHECKPOINT_FILE = "XML_Roberta_neural_network_Anonimizator_finetune/labeling_checkpoint.txt"
    SAVE_INTERVAL = 10  # Save progress every N resumes
    
    print("=" * 60)
    print("DATASET ENTITY LABELER")
    print("=" * 60)
    
    # Load resumes
    print(f"\n📂 Loading resumes from: {INPUT_FILE}")
    resumes = load_resumes(INPUT_FILE)
    print(f"✅ Loaded {len(resumes)} resumes")
    
    # Initialize labeler
    print("\n🤖 Initializing ChatGPT API...")
    labeler = DatasetLabeler()
    print("✅ API initialized")
    
    # Load checkpoint
    start_index = load_checkpoint(CHECKPOINT_FILE)
    if start_index > 0:
        print(f"\n🔄 Resuming from resume #{start_index}")
    
    # Load existing labeled data if resuming
    labeled_data = []
    if start_index > 0 and os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing_text = f.read()
            labeled_data = [r.strip() for r in existing_text.split('\n\n') if r.strip()]
        print(f"✅ Loaded {len(labeled_data)} existing labeled resumes")
    
    # Process resumes
    print(f"\n🏃 Processing resumes {start_index + 1} to {len(resumes)}...")
    print("-" * 60)
    
    failed_resumes = []
    
    for i in tqdm(range(start_index, len(resumes)), desc="Labeling"):
        resume = resumes[i]
        
        try:
            # Extract entities
            entities = labeler.extract_entities(resume)
            
            # Convert to CoNLL format
            token_tag_pairs = labeler.entities_to_conll(resume, entities)
            conll_output = labeler.format_conll_output(token_tag_pairs)
            
            # Add to labeled data
            labeled_data.append(conll_output)
            
            # Save progress periodically
            if (i + 1) % SAVE_INTERVAL == 0:
                save_progress(OUTPUT_FILE, labeled_data)
                save_checkpoint(CHECKPOINT_FILE, i + 1)
            
            # Small delay to avoid rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\n❌ Error processing resume #{i}: {e}")
            failed_resumes.append(i)
            # Save a placeholder for failed resume
            labeled_data.append(f"# ERROR: Resume {i} failed to process\n")
            continue
    
    # Final save
    print("\n\n💾 Saving final results...")
    save_progress(OUTPUT_FILE, labeled_data)
    save_checkpoint(CHECKPOINT_FILE, len(resumes))
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ LABELING COMPLETE")
    print("=" * 60)
    print(f"Total resumes: {len(resumes)}")
    print(f"Successfully labeled: {len(resumes) - len(failed_resumes)}")
    print(f"Failed: {len(failed_resumes)}")
    print(f"\nOutput saved to: {OUTPUT_FILE}")
    
    if failed_resumes:
        print(f"\nFailed resume indices: {failed_resumes[:20]}")
        if len(failed_resumes) > 20:
            print(f"... and {len(failed_resumes) - 20} more")
    
    # Cleanup checkpoint on success
    if not failed_resumes:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("\n🗑️ Checkpoint file removed")


if __name__ == "__main__":
    main()
