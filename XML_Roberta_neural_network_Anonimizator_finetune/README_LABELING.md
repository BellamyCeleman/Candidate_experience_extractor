# Dataset Entity Labeling Guide

This directory contains scripts to label entities in resume datasets using ChatGPT API for Named Entity Recognition (NER) training.

## Entity Types

The labeling system identifies 6 entity types:

1. **DATES** (`DATE`) - Any dates, periods, years
   - Examples: "2020-2023", "January 2021", "OCT 2021", "15.02.1982"

2. **COMPANIES** (`ORG`) - Names of companies/organizations
   - Examples: "Google", "Microsoft", "BitAlpha"

3. **HARD_SKILLS** (`SKILL`) - Technical skills only (NO soft skills)
   - Examples: "Python", "SQL", "Docker", "Tableau", "AWS"
   - NOT: "communication", "teamwork", "leadership"

4. **FULL_NAME** (`PER`) - Person's full name
   - Examples: "John Smith", "Maria Garcia Lopez"

5. **PROFESSION** (`PROF`) - Job titles and professions
   - Examples: "Python Developer", "Business Analyst", "Senior Engineer"

6. **O** - All other tokens (outside entities)

## Output Format

CoNLL format with BIO tagging:
```
Token B-TAG/I-TAG/O
Token B-TAG/I-TAG/O
...
```

- `B-` = Begin (first token of entity)
- `I-` = Inside (continuation of entity)
- `O` = Outside (not an entity)

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
Skills O
: O
Python B-SKILL
, O
SQL B-SKILL
```

## Scripts

### 1. `test_labeler.py` - Test on Sample

Tests the labeling pipeline on a single sample resume.

**Usage:**
```bash
python XML_Roberta_neural_network_Anonimizator_finetune/test_labeler.py
```

**Output:**
- Displays extracted entities
- Shows CoNLL format output
- Saves to `sample_labeled.conll`

### 2. `dataset_labeler.py` - Full Dataset Labeling

Labels the entire dataset using ChatGPT API.

**Features:**
- ✅ Progress tracking with tqdm
- ✅ Checkpoint/resume capability
- ✅ Error handling and retry logic
- ✅ Rate limit protection
- ✅ Periodic saves every 10 resumes

**Usage:**
```bash
python XML_Roberta_neural_network_Anonimizator_finetune/dataset_labeler.py
```

**Input:** `datasets/dataset_labeling.txt` (1,006 resumes)
**Output:** `XML_Roberta_neural_network_Anonimizator_finetune/dataset_labeled.conll`

**Checkpoint:** `labeling_checkpoint.txt` (auto-saves progress)

**To resume after interruption:**
Just run the script again - it will automatically continue from where it left off.

## Configuration

The scripts use Azure OpenAI API configured in `ChatGPT/config.py`:

- Model: `gpt-4o-mini`
- Temperature: 0.1 (deterministic)
- Max tokens: 4096

### Environment Variables

Required in `dev.env`:
```
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
```

## Workflow

### Quick Test (Recommended First Step)

1. Test on sample to verify API and labeling:
```bash
python XML_Roberta_neural_network_Anonimizator_finetune/test_labeler.py
```

2. Review the output in `sample_labeled.conll`

3. Check that entities are correctly identified and tagged

### Full Dataset Labeling

1. Ensure `dev.env` has correct API credentials

2. Run the labeler:
```bash
python XML_Roberta_neural_network_Anonimizator_finetune/dataset_labeler.py
```

3. Monitor progress (will show progress bar)

4. If interrupted, just run again to resume

5. Output will be saved to `dataset_labeled.conll`

## Expected Runtime

- **Sample test:** ~10-15 seconds
- **Full dataset (1,006 resumes):**
  - ~0.5 seconds per resume
  - Total: ~8-10 minutes (with rate limits)
  - Cost: ~$0.10-0.20 (depending on API pricing)

## Troubleshooting

### Rate Limit Errors
The script automatically handles rate limits with exponential backoff.

### JSON Parse Errors
Script retries up to 3 times per resume. If persistent, check API response format.

### Content Filter Errors
Some resume text may trigger content filters. These will be skipped with error messages.

### Resume from Checkpoint
If the script crashes or is interrupted:
```bash
# Just run again - it will automatically resume
python XML_Roberta_neural_network_Anonimizator_finetune/dataset_labeler.py
```

### Clear Progress and Start Over
```bash
# Delete checkpoint file
del XML_Roberta_neural_network_Anonimizator_finetune\labeling_checkpoint.txt

# Delete partial output (optional)
del XML_Roberta_neural_network_Anonimizator_finetune\dataset_labeled.conll

# Run again
python XML_Roberta_neural_network_Anonimizator_finetune/dataset_labeler.py
```

## Next Steps

After labeling is complete:

1. **Review the output** - Check `dataset_labeled.conll` for quality

2. **Split dataset** - Use `data_loader.py` to split into train/val:
```python
from data_loader import split_and_save_data
split_and_save_data(
    "XML_Roberta_neural_network_Anonimizator_finetune/dataset_labeled.conll",
    "XML_Roberta_neural_network_Anonimizator_finetune/train.txt",
    "XML_Roberta_neural_network_Anonimizator_finetune/val.txt"
)
```

3. **Train model** - Run the fine-tuning pipeline:
```bash
python XML_Roberta_neural_network_Anonimizator_finetune/finetune_pipeline.py
```

4. **Test model** - Evaluate the trained model:
```bash
python XML_Roberta_neural_network_Anonimizator_finetune/test.py
```

## File Structure

```
XML_Roberta_neural_network_Anonimizator_finetune/
├── dataset_labeler.py          # Main labeling script
├── test_labeler.py             # Sample test script
├── README_LABELING.md          # This file
├── labeling_checkpoint.txt     # Progress checkpoint (auto-generated)
├── sample_labeled.conll        # Sample output (from test)
└── dataset_labeled.conll       # Full labeled dataset (output)
```

## Quality Assurance

After labeling, verify:

1. ✅ All resumes processed (check count)
2. ✅ Entities properly tagged (B- and I- tags)
3. ✅ No empty entity spans
4. ✅ Token-tag alignment correct
5. ✅ Entity types match expected categories

```python
# Quick quality check
with open('dataset_labeled.conll', 'r') as f:
    lines = f.readlines()
    entity_lines = [l for l in lines if not l.startswith('O') and l.strip()]
    print(f"Total entity tokens: {len(entity_lines)}")
```
