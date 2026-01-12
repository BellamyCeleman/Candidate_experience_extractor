"""
Test script for dataset labeler
Tests the labeling pipeline on a few sample resumes
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_labeler import DatasetLabeler


def test_sample_resume():
    """Test labeling on a sample resume"""
    
    sample_resume = """MYKHAILO KAKORIN
Vienna, Austria
15.02.1982
mkakorin@gmail.com

SUMMARY
A versatile individual with experience in data analysis. Excellent understanding and proficiency in SQL, Tableau, Looker, and R.

EXPERIENCE
BitAlpha, Kyiv/Vienna — Business Analyst
OCT 2021 - DEC 2022
● Analyze and interpret data to identify business needs
● Developed reports using Tableau and SQL
● Collaborated with stakeholders to improve processes

SKILLS
- Data Analysis
- SQL, Python, R
- Tableau, Looker
- Process Improvement"""

    print("=" * 60)
    print("TESTING DATASET LABELER")
    print("=" * 60)
    
    # Initialize labeler
    print("\n🤖 Initializing ChatGPT API...")
    labeler = DatasetLabeler()
    print("✅ API initialized")
    
    # Extract entities
    print("\n🔍 Extracting entities...")
    entities = labeler.extract_entities(sample_resume)
    
    print(f"\n✅ Found {len(entities)} entities:")
    print("-" * 60)
    
    # Group by type
    by_type = {}
    for e in entities:
        if e.type not in by_type:
            by_type[e.type] = []
        by_type[e.type].append(e)
    
    for entity_type, items in sorted(by_type.items()):
        print(f"\n{entity_type}:")
        for e in items:
            print(f"  [{e.start}:{e.end}] '{e.text}'")
    
    # Convert to CoNLL format
    print("\n" + "=" * 60)
    print("CoNLL FORMAT OUTPUT")
    print("=" * 60)
    
    token_tag_pairs = labeler.entities_to_conll(sample_resume, entities)
    conll_output = labeler.format_conll_output(token_tag_pairs)
    
    print(conll_output)
    
    # Save sample output
    output_file = "XML_Roberta_neural_network_Anonimizator_finetune/sample_labeled.conll"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(conll_output)
    
    print("\n" + "=" * 60)
    print(f"✅ Sample output saved to: {output_file}")
    print("=" * 60)
    
    # Statistics
    total_tokens = len(token_tag_pairs)
    entity_tokens = sum(1 for _, tag in token_tag_pairs if tag != 'O')
    
    print(f"\nStatistics:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Entity tokens: {entity_tokens}")
    print(f"  Non-entity tokens: {total_tokens - entity_tokens}")
    print(f"  Entity ratio: {entity_tokens / total_tokens * 100:.1f}%")


if __name__ == "__main__":
    test_sample_resume()
