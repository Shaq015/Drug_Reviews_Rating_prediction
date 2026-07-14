# Drug_Reviews_Rating_prediction

This project predicts patient drug review ratings from text reviews, drug names, and medical conditions.  
The task is based on the Drugs.com review dataset and includes both rating prediction and GenAI-based summary/sentiment generation.

## Task

Given a patient review, drug name, condition, and true rating, the goal is to predict the 1-10 patient rating.  
For each test review, a generative AI model is also used to generate:

- a summary of up to 10 words
- a binary sentiment label: positive or negative

## Approach

The project includes:

- EDA and preprocessing
- classical baselines
- deep learning models
- transformer fine-tuning
- ordinal-learning experiments
- GenAI summary and sentiment generation

## Main Models

The final selected models were:

1. Multi-Head BERT  
2. Soft Labels BERT

## Results

| Model | Accuracy | Macro-F1 | MAE | Within-1 |
|---|---:|---:|---:|---:|
| Multi-Head BERT | 0.630 | 0.518 | 0.667 | 0.857 |
| Soft Labels BERT | 0.564 | 0.438 | 0.728 | 0.853 |

## GenAI Component

Qwen2.5-7B-Instruct was used to generate short summaries and sentiment labels.  
Several prompting strategies were tested, and the medical-focused prompt was selected for the final generation.
