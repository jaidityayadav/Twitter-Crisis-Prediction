
# Twitter Crisis Prediction

This project classifies tweets as "crisis" or "not a crisis" using a RoBERTa-based NLP model. It also leverages Named Entity Recognition (NER) to extract location information, which helps improve crisis detection.

## Project Overview

- **Task:** Given a CSV of tweets (`train.csv`), predict whether each tweet describes a crisis.
- **Model:** RoBERTa transformer for sequence classification.
- **Feature Engineering:** Includes NER for location extraction, sentiment analysis, and text-based features.

## Workflow

1. **Data Loading & Exploration**
	- Load `train.csv` into a pandas DataFrame.
	- Explore target distribution, tweet lengths, top locations, and keywords.

2. **Data Cleaning**
	- Remove missing values and duplicates.
	- Clean tweet text (remove URLs, mentions, special characters).

3. **Feature Engineering**
	- Use spaCy for NER to extract location entities.
	- Add features: location presence, word count, sentiment polarity/subjectivity.

4. **Data Splitting**
	- Split data into training, validation, and test sets (stratified by target).

5. **Model Training**
	- Tokenize tweets using RoBERTa tokenizer.
	- Train RoBERTa for binary classification.
	- Evaluate using accuracy, precision, recall, and F1 score.

6. **Model Optimization**
	- Use Optuna for hyperparameter tuning.
	- Retrain and save the best model.

## Requirements

- Python
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- spaCy (`en_core_web_sm` model)
- textblob
- transformers (HuggingFace)
- optuna

## Usage

1. Place `train.csv` in the project directory.
2. Run the notebook `roberta.ipynb` step by step.
3. The trained and optimized models are saved as `roberta_model` and `optimized_roberta_model`.

## Notes

- Location features help identify crisis tweets related to specific places.
- The pipeline is modular and can be extended with more features or models.
