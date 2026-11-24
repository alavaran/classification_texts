# Financial News Sentiment Classification

## Overview

A machine learning project for classifying financial news sentences by sentiment.

The model predicts one of three classes:

- **Negative** — negative financial or business sentiment
- **Neutral** — neutral or factual information
- **Positive** — positive financial or business sentiment

The project covers the end-to-end text classification workflow: data exploration, preprocessing, model training, evaluation, error analysis, model serialization, and inference on new texts.

## Dataset

**Financial PhraseBank — 75% agreement subset**

- Source: [Financial PhraseBank on Hugging Face](https://huggingface.co/datasets/takala/financial_phrasebank)
- Initial dataset size: 3,453 sentences
- Cleaned dataset size: 3,448 sentences
- Target classes: Negative, Neutral, Positive

Class distribution after cleaning:

| Sentiment | Samples | Share |
|---|---:|---:|
| Negative | 420 | 12.2% |
| Neutral | 2,141 | 62.1% |
| Positive | 887 | 25.7% |

## Methodology

### Preprocessing

The following text preprocessing steps were applied:

- Lowercase conversion
- HTML tag removal
- Whitespace normalization

Punctuation and stop words were preserved.

### Baseline

The initial baseline used:

- TF-IDF vectorization
- Unigram and bigram features
- Logistic Regression
- No class weighting

Test metrics:

| Metric | Score |
|---|---:|
| Accuracy | 0.826 |
| Macro F1 | 0.748 |

### Model comparison and improvement

Several experiments were conducted:

- Logistic Regression with class weighting
- TF-IDF with different n-gram ranges
- LinearSVC with class weighting
- Logistic Regression with different regularization strengths

The best validation result was achieved by **LinearSVC** with balanced class weights.

Final model configuration:

- TF-IDF: `ngram_range=(1, 2)`
- `max_features=10000`
- `min_df=2`
- `max_df=0.95`
- `sublinear_tf=True`
- LinearSVC: `C=1.0`
- `class_weight="balanced"`

### Final test results

| Metric | Score |
|---|---:|
| Accuracy | 0.868 |
| Macro Precision | 0.87 |
| Macro Recall | 0.79 |
| Macro F1 | 0.824 |
| Weighted F1 | 0.86 |

Per-class F1:

| Class | F1-score |
|---|---:|
| Negative | 0.77 |
| Neutral | 0.91 |
| Positive | 0.79 |

The final model improved Macro F1 from 0.748 to 0.824 compared with the initial baseline.

## Error Analysis

The final model produced 91 errors on 690 test samples.

The most frequent error types were:

| Actual → Predicted | Errors |
|---|---:|
| Positive → Neutral | 40 |
| Neutral → Positive | 18 |
| Negative → Neutral | 17 |
| Negative → Positive | 11 |
| Positive → Negative | 4 |
| Neutral → Negative | 1 |

Main limitations identified:

- Positive and negative financial signals are sometimes classified as neutral.
- The model does not reliably interpret financial context.
- It may confuse a decrease in costs with a negative business event.
- It has limited ability to interpret numerical comparisons and business dynamics.

Potential improvements include domain-specific language models such as FinBERT, contextual embeddings, and additional analysis of ambiguous training labels.

## Inference

The trained model and TF-IDF vectorizer are saved in the `models/` directory.

To run inference:

```bash
python inference.py
```

The inference pipeline:

1. Loads the saved model and vectorizer.
2. Preprocesses the input text.
3. Converts the text into TF-IDF features.
4. Predicts the sentiment class.
5. Returns the original text, processed text, and predicted sentiment.

Example:

```python
from inference import predict_sentiment

result = predict_sentiment(
    "The company reported strong revenue growth."
)

print(result["sentiment"])
# Positive
```

## Installation

Clone the repository and create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run the training script:

```bash
python main.py
```

The script trains the model and saves the model artifacts in `models/`.

## Project Structure

```text
classification_day07/
├── main.py
├── inference.py
├── README.md
├── requirements.txt
├── .gitignore
├── models/
├── reports/
└── .venv/
```

## Limitations

This project is an educational financial sentiment classification baseline, not a production-grade financial analysis system.

The model:

- Is trained on a relatively small dataset.
- Uses TF-IDF and a linear classifier rather than contextual language representations.
- Does not understand financial events in the same way as a human analyst.
- Does not provide calibrated prediction probabilities.
- May produce incorrect predictions on unfamiliar financial terminology or ambiguous statements.

Predictions should not be used as the sole basis for financial or investment decisions.

## Future Improvements

- Benchmark against FinBERT.
- Add automated tests with pytest.
- Introduce a FastAPI inference endpoint.
- Add confidence analysis and monitoring.
- Improve handling of numerical and context-dependent financial expressions.

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- TF-IDF
- Logistic Regression
- LinearSVC
- joblib
