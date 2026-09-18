# Sawyer-RLHF: A Controlled Low-Resource Classical RLHF Baseline on GPT-2 (124M)

**A fully reproducible end-to-end SFT → Reward Model → PPO pipeline under single-GPU constraints.**

This repository contains a complete, minimal, and transparent classical RLHF experiment performed on `openai-community/gpt2` (124M parameters) using a single NVIDIA T4 (16 GB). The goal is to provide a carefully controlled baseline that quantifies the gap between proxy reward and independent quality under severe resource limitations.

## Key Results (Actual Run)

| Stage              | Metric                          | Value          |
|--------------------|---------------------------------|----------------|
| SFT                | Final validation loss           | 2.695          |
| Reward Model       | Validation preference accuracy  | 55.3 %         |
| PPO                | Mean reward difference (PPO−SFT)| +0.145         |
| Independent Judge  | SFT preferred                   | 60 %           |
| Independent Judge  | PPO preferred                   | 40 %           |
| Over-optimization  | Cases where reward ↑ but judge preferred SFT | 9 / 25 |

Despite a modest positive reward-model signal, independent quality judgments preferred the SFT policy. This constitutes a clear demonstration of **reward over-optimization** under classical RLHF when both policy and reward model are capacity-constrained.

## Pipeline Overview

1. **Supervised Fine-Tuning** on Databricks Dolly-15k (2 epochs)
2. **Reward Model** (Bradley-Terry) on Anthropic HH-RLHF helpful-base (2 epochs)
3. **PPO** for 300 updates with β = 0.02
4. **Independent Evaluation** on 100 held-out prompts + AI judge on 25 examples

## Hardware & Software

- Single NVIDIA T4 (16 GB) or free Google Colab T4
- Transformers + TRL 0.11.4
- fp16 + gradient checkpointing throughout

## Quick Start

```bash
pip install -r requirements.txt
