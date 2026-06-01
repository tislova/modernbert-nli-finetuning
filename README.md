# ModernBERT NLI Fine-Tuning & Error Analysis

Fine-tuning ModernBERT-base on the AllNLI dataset for natural language 
inference, with a focus on understanding where the model fails rather than 
just reporting accuracy. The project includes per-class error analysis, 
inspection of specific failure cases, and a small set of probing experiments 
to test hypotheses about model behavior.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗_transformers-FFD21E?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

## Overview

Natural Language Inference (NLI) is the task of deciding whether a hypothesis 
sentence is entailed by, contradicts, or is neutral with respect to a premise 
sentence. It's a useful benchmark for testing how well a model handles meaning 
rather than surface-level patterns.

This project fine-tunes ModernBERT-base, Answer.AI's 2024 encoder model, on 
a sample of the AllNLI dataset, then examines its behavior in detail: which 
class it struggles with most, what kinds of inputs trip it up, and whether a 
few targeted hypotheses about its failure modes hold up under manually 
constructed test cases.

## Dataset & Model

- **Dataset:** [`sentence-transformers/all-nli`](https://huggingface.co/datasets/sentence-transformers/all-nli), pair-class subset (3% sample for compute feasibility)
- **Model:** [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base)
- **Task:** 3-class classification — entailment, neutral, contradiction
- **Input format:** premise and hypothesis joined with a `<s>` separator
- **Split:** 90% train / 10% test

## Training Setup

Two epochs, batch size 4 for training and 2 for evaluation, learning rate 
5e-5. Training used `bf16` mixed precision with the fused AdamW optimizer, 
evaluated every 500 steps, and kept the best checkpoint by eval loss. Maximum 
sequence length was 500 tokens.

## Results

Overall accuracy on the held-out test set was **83.55%**: 2,827 correct 
predictions out of 3,292.

The more informative view is per-class:

| Class | TP | FP | TN | FN | Precision |
|-------|----|----|----|----|-----------|
| entailment | 838 | 169 | 1,722 | 98 | 0.83 |
| neutral | 749 | 195 | 1,678 | 205 | **0.79** |
| contradiction | 775 | 101 | 1,789 | 162 | **0.88** |

Neutral is consistently the hardest class. It has the most false positives, 
the most false negatives, and the lowest precision. Contradiction is the 
model's strongest category, with the highest precision and the fewest false 
positives by a clear margin. This pattern makes sense intuitively: 
contradictions tend to involve sharper lexical signals (negation, opposing 
facts), while neutral cases require the model to recognize the *absence* of 
either entailment or contradiction, which is a subtler judgment.
