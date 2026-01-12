# Quick Start Guide - Entity Labeling

## Overview

Automated entity labeling system for resume datasets using ChatGPT API. Converts raw text into CoNLL format for NER model training.

## Quick Test (2 minutes)

```bash
cd d:\PythonProjects\Candidate_exp_extractor\Candidate_experience_extractor
python XML_Roberta_neural_network_Anonimizator_finetune/test_labeler.py
```

Expected output:
- ✅ Entities extracted (names, dates, companies, skills, professions)
- ✅ CoNLL format output with BIO tags
- ✅ Sample saved to `sample_labeled.conll`

## Full Dataset Labeling (~10 minutes)

```bash
python XML_Roberta_neural_network_Anonimizator_finetune/dataset_labeler.py
```

This will:
- Process 1,006 resumes from `datasets/dataset_labeling.txt`
- Save to `XML_Roberta_neural_network_Anonimizator_finetune/dataset_labeled.conll`
- Show progress bar
- Auto-save every 10 resumes (checkpoint)
- Handle errors and rate limits automatically

### If Interrupted

Just run the same command again - it will automatically resume from where it stopped!

## Verify Output

```python
# Check the output
with open('XML_Roberta_neural_network_Anonimizator_finetune/dataset_labeled.conll', 'r', encoding='utf-8') as f:
    content = f.read()
    resumes = content.split('\n\n')
    print(f"Labeled resumes: {len(resumes)}")
    
    # Count entity tags
    lines = content.split('\n')
    entity_lines = [l for l in lines if l.strip() and not l.strip().endswith(' O')]
    print(f"Entity tokens: {len(entity_lines)}")
```

## Entity Types

| Entity Type | Tag | Example |
|------------|-----|---------|
| Full Name | PER | `MYKHAILO` B-PER, `KAKORIN` I-PER |
| Dates | DATE | `OCT` B-DATE, `2021` I-DATE |
| Companies | ORG | `BitAlpha` B-ORG |
| Skills | SKILL | `Python` B-SKILL, `SQL` B-SKILL |
| Profession | PROF | `Business` B-PROF, `Analyst` I-PROF |
| Other | O | All other words |

## Output Format

CoNLL format with BIO tagging:

```
Token TAG
Token TAG
Token TAG

(blank line between resumes)
```

Example:
```
MYKHAILO B-PER
KAKORIN I-PER
Senior B-PROF
Python I-PROF
Developer I-PROF
at O
Google B-ORG
2020 B-DATE
- O
2023 B-DATE
```

## Troubleshooting

### API Key Error
Check `dev.env` has:
```
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
```

### Rate Limit
Script handles automatically with exponential backoff. Just wait.

### Resume from Checkpoint
Delete checkpoint to start over:
```bash
del XML_Roberta_neural_network_Anonimizator_finetune\labeling_checkpoint.txt
```

## Next Steps

After labeling completes:

1. **Split dataset** into train/val:
```bash
python XML_Roberta_neural_network_Anonimizator_finetune/finetune_pipeline.py
```
(Pipeline automatically splits if train.txt/val.txt don't exist)

2. **Train model** - Same command as above, it will train after splitting

3. **Test model**:
```bash
python XML_Roberta_neural_network_Anonimizator_finetune/test.py
```

## File Locations

- **Input:** `datasets/dataset_labeling.txt` (raw text, 1,006 resumes)
- **Output:** `XML_Roberta_neural_network_Anonimizator_finetune/dataset_labeled.conll`
- **Checkpoint:** `XML_Roberta_neural_network_Anonimizator_finetune/labeling_checkpoint.txt`
- **Sample:** `XML_Roberta_neural_network_Anonimizator_finetune/sample_labeled.conll`

## Cost Estimate

- Model: `gpt-4o-mini`
- Resumes: 1,006
- Avg tokens per resume: ~800 input + ~300 output
- Estimated cost: **$0.10 - $0.20**
- Time: **8-12 minutes**
