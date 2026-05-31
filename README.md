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
