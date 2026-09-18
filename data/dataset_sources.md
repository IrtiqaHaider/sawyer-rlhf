## Supervised Fine-Tuning
- **Name**: Databricks Dolly-15k
- **Hugging Face ID**: `databricks/databricks-dolly-15k`
- **Split used**: train (90/10 train/val)
- **License**: CC BY-SA 3.0

## Reward Model
- **Name**: Anthropic HH-RLHF (helpful-base)
- **Hugging Face ID**: `Anthropic/hh-rlhf`
- **Subset**: helpful-base
- **Pairs used**: 12 000 (shuffled, seed=42)
- **License**: MIT

## PPO Prompts
- Subset of Dolly-15k instructions (6 000 examples)

## Evaluation
- 100 held-out prompts from Dolly-15k (final 100 after seed=42 shuffle)