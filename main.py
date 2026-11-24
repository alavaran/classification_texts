import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
import os


def preprocess_text(text: str) -> str:
    """
    Preprocess a financial news sentence.
    """

    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


url = (
    "https://huggingface.co/datasets/takala/"
    "financial_phrasebank/resolve/refs%2Fpr%2F10/"
    "sentences_75agree/train-00000-of-00001.parquet"
)

df = pd.read_parquet(url)

df = df.rename(columns={
    "sentence": "text",
    "label": "sentiment"
})

print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.duplicated().sum())
print(df["sentiment"].value_counts())

duplicates = df[df.duplicated(keep=False)]

print(duplicates.sort_values("text"))

print(df.groupby("sentiment")["text"].count())
print(df["sentiment"].unique())

label_mapping = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

df["char_count"] = df["text"].str.len()
df["word_count"] = df["text"].str.split().str.len()

print(df[["char_count", "word_count"]].describe())

sentiment_stats = (
    df.groupby("sentiment")[["char_count", "word_count"]]
    .agg(["mean", "median", "min", "max"])
)

print(sentiment_stats)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(
    data=df,
    x="char_count",
    bins=40,
    kde=True,
    ax=axes[0]
)

sns.histplot(
    data=df,
    x="word_count",
    bins=40,
    kde=True,
    ax=axes[1]
)

axes[0].set_title("Character Count Distribution")
axes[1].set_title("Word Count Distribution")

plt.tight_layout()
plt.savefig("reports/text_length_distribution.png", dpi=150)
plt.show()

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="sentiment",
    y="word_count"
)

plt.title("Text Length by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Word Count")

plt.savefig("reports/word_count_by_sentiment.png", dpi=150)
plt.show()
plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="sentiment",
    y="word_count"
)

plt.title("Text Length by Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Word Count")

plt.savefig("reports/word_count_by_sentiment.png", dpi=150)
plt.show()

for label, name in {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}.items():

    print(f"\n===== {name} =====")

    samples = df[df["sentiment"] == label]["text"].sample(
        5,
        random_state=42
    )

    for text in samples:
        print("-", text)

# Создаем рабочую копию
df_clean = df.copy()

# Удаляем точные дубликаты
df_clean = df_clean.drop_duplicates(subset=["text"])

# Удаляем строки с отсутствующим текстом или меткой
df_clean = df_clean.dropna(subset=["text", "sentiment"])

# Удаляем пустые строки и строки только из пробелов
df_clean = df_clean[df_clean["text"].str.strip() != ""]

print("Original shape:", df.shape)
print("Cleaned shape:", df_clean.shape)

def preprocess_text(text):
    """
    Basic text preprocessing for financial news.
    """
    if not isinstance(text, str):
        return ""

    # Приводим к нижнему регистру
    text = text.lower()

    # Удаляем HTML-теги
    text = re.sub(r"<[^>]+>", " ", text)

    # Убираем лишние пробелы, табуляции и переносы строк
    text = re.sub(r"\s+", " ", text)

    # Убираем пробелы по краям
    text = text.strip()

    return text

df_clean["processed_text"] = df_clean["text"].apply(preprocess_text)

# Проверяем результат
print(df_clean[["text", "processed_text"]].head(10))

empty_texts = (df_clean["processed_text"].str.len() == 0).sum()

print("Empty processed texts:", empty_texts)

df_clean = df_clean[df_clean["processed_text"].str.len() > 0]

label_mapping = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

df_clean["sentiment_name"] = df_clean["sentiment"].map(label_mapping)

print(df_clean[["sentiment", "sentiment_name"]].head())
print(df_clean["sentiment_name"].value_counts())

test_samples = [
    "THE COMPANY REPORTED A 25% INCREASE IN REVENUE.",
    "Profit declined by 15% in the third quarter.",
    "The company reported stable financial performance.",
    "   Revenue   increased   significantly.   "
]

for sample in test_samples:
    print("Original:", sample)
    print("Processed:", preprocess_text(sample))
    print("-" * 60)

print("Dataset shape:", df_clean.shape)

print("\nMissing values:")
print(df_clean[["processed_text", "sentiment"]].isnull().sum())

print("\nDuplicate processed texts:")
print(df_clean["processed_text"].duplicated().sum())

print("\nClass distribution:")
print(df_clean["sentiment_name"].value_counts())


from sklearn.model_selection import train_test_split

X = df_clean["processed_text"]
y = df_clean["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

print("\nTrain distribution:")
print(y_train.value_counts(normalize=True))

print("\nTest distribution:")
print(y_test.value_counts(normalize=True))


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Train TF-IDF shape:", X_train_tfidf.shape)
print("Test TF-IDF shape:", X_test_tfidf.shape)

print("Vocabulary size:", len(tfidf.vocabulary_))


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight=None
)

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Model trained successfully!")
print("Predictions:", y_pred[:10])
print("Actual:", y_test.iloc[:10].values)


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, average="macro"
)

recall = recall_score(
    y_test, y_pred, average="macro"
)

f1 = f1_score(
    y_test, y_pred, average="macro"
)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Negative", "Neutral", "Positive"]
))


import pandas as pd
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

cm_df = pd.DataFrame(
    cm,
    index=["Actual Negative", "Actual Neutral", "Actual Positive"],
    columns=["Pred Negative", "Pred Neutral", "Pred Positive"]
)

print(cm_df)


model_balanced = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

model_balanced.fit(X_train_tfidf, y_train)

y_pred_balanced = model_balanced.predict(X_test_tfidf)

print("Balanced model trained!")


from sklearn.metrics import classification_report, accuracy_score, f1_score

print("Accuracy:", accuracy_score(y_test, y_pred_balanced))

print("Macro F1:", f1_score(
    y_test,
    y_pred_balanced,
    average="macro"
))

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred_balanced,
    labels=[0, 1, 2],
    target_names=["Negative", "Neutral", "Positive"]
))


cm_balanced = confusion_matrix(
    y_test,
    y_pred_balanced,
    labels=[0, 1, 2]
)

sns.heatmap(
    cm_balanced,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Neutral", "Positive"],
    yticklabels=["Negative", "Neutral", "Positive"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Balanced Logistic Regression")
plt.tight_layout()
plt.show()


import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

results = {
    "experiment_01_baseline": {
        "vectorizer": "TF-IDF",
        "classifier": "LogisticRegression",
        "class_weight": None,
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(
            y_test, y_pred, average="macro"
        ),
        "macro_recall": recall_score(
            y_test, y_pred, average="macro"
        ),
        "macro_f1": f1_score(
            y_test, y_pred, average="macro"
        )
    },
    "experiment_02_balanced": {
        "vectorizer": "TF-IDF",
        "classifier": "LogisticRegression",
        "class_weight": "balanced",
        "accuracy": accuracy_score(y_test, y_pred_balanced),
        "macro_precision": precision_score(
            y_test, y_pred_balanced, average="macro"
        ),
        "macro_recall": recall_score(
            y_test, y_pred_balanced, average="macro"
        ),
        "macro_f1": f1_score(
            y_test, y_pred_balanced, average="macro"
        )
    }
}

with open("baseline_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("Results saved successfully!")
from sklearn.model_selection import train_test_split

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

print("Fit size:", len(X_fit))
print("Validation size:", len(X_val))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# TF-IDF with trigrams
tfidf_trigram = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_fit_trigram = tfidf_trigram.fit_transform(X_fit)
X_val_trigram = tfidf_trigram.transform(X_val)

# Balanced Logistic Regression
model_trigram = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

model_trigram.fit(X_fit_trigram, y_fit)

y_pred_trigram = model_trigram.predict(X_val_trigram)

print("Vocabulary size:", len(tfidf_trigram.vocabulary_))
print("Accuracy:", accuracy_score(y_val, y_pred_trigram))
print("Macro F1:", f1_score(
    y_val,
    y_pred_trigram,
    average="macro"
))

print("\nClassification Report:")
print(classification_report(
    y_val,
    y_pred_trigram,
    labels=[0, 1, 2],
    target_names=["Negative", "Neutral", "Positive"]
))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# Original TF-IDF configuration
tfidf_svc = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_fit_svc = tfidf_svc.fit_transform(X_fit)
X_val_svc = tfidf_svc.transform(X_val)

# LinearSVC
model_svc = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=42,
    max_iter=2000
)

model_svc.fit(X_fit_svc, y_fit)

y_pred_svc = model_svc.predict(X_val_svc)

print("Vocabulary size:", len(tfidf_svc.vocabulary_))
print("Accuracy:", accuracy_score(y_val, y_pred_svc))
print("Macro F1:", f1_score(
    y_val,
    y_pred_svc,
    average="macro"
))

print("\nClassification Report:")
print(classification_report(
    y_val,
    y_pred_svc,
    labels=[0, 1, 2],
    target_names=["Negative", "Neutral", "Positive"]
))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

C_values = [0.1, 0.5, 1.0, 2.0, 5.0]

tuning_results = []

for c in C_values:
    lr = LogisticRegression(
        C=c,
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )

    lr.fit(X_fit_svc, y_fit)
    pred = lr.predict(X_val_svc)

    tuning_results.append({
        "C": c,
        "accuracy": accuracy_score(y_val, pred),
        "macro_f1": f1_score(y_val, pred, average="macro")
    })

results_df = pd.DataFrame(tuning_results)

print(results_df.sort_values("macro_f1", ascending=False))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

baseline_results = []

for name, weight in [
    ("LR baseline", None),
    ("LR balanced", "balanced")
]:
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight=weight
    )

    model.fit(X_fit_svc, y_fit)
    pred = model.predict(X_val_svc)

    baseline_results.append({
        "experiment": name,
        "accuracy": accuracy_score(y_val, pred),
        "macro_f1": f1_score(y_val, pred, average="macro")
    })

print(pd.DataFrame(baseline_results))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# TF-IDF на полном train
tfidf_final = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_final = tfidf_final.fit_transform(X_train)
X_test_final = tfidf_final.transform(X_test)

# Финальная модель
final_model = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=42,
    max_iter=2000
)

final_model.fit(X_train_final, y_train)

# Предсказания
y_pred_final = final_model.predict(X_test_final)

print("Accuracy:", accuracy_score(y_test, y_pred_final))
print("Macro F1:", f1_score(y_test, y_pred_final, average="macro"))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_final,
    target_names=["Negative", "Neutral", "Positive"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))

# ==========================================
# DAY 5 — ERROR ANALYSIS
# ==========================================

# 1. Создаем DataFrame с предсказаниями

label_mapping = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

error_df = pd.DataFrame({
    "text": X_test.to_numpy(),
    "true_label": y_test.to_numpy(),
    "predicted_label": y_pred_final
})

error_df["true_sentiment"] = error_df["true_label"].map(label_mapping)
error_df["predicted_sentiment"] = error_df["predicted_label"].map(label_mapping)

# 2. Выделяем ошибочные предсказания

errors = error_df[
    error_df["true_label"] != error_df["predicted_label"]
].copy()

print("\n===== ERROR ANALYSIS =====")

print("Total samples:", len(error_df))
print("Correct predictions:", len(error_df) - len(errors))
print("Total errors:", len(errors))
print("Error rate:", round(len(errors) / len(error_df), 4))

print("\nErrors by actual class:")
print(errors["true_sentiment"].value_counts())

print("\nErrors by predicted class:")
print(errors["predicted_sentiment"].value_counts())

import os
from sklearn.metrics import confusion_matrix

os.makedirs("reports", exist_ok=True)

labels = [0, 1, 2]
class_names = ["Negative", "Neutral", "Positive"]

cm = confusion_matrix(
    y_test,
    y_pred_final,
    labels=labels
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted sentiment")
plt.ylabel("Actual sentiment")
plt.title("LinearSVC — Confusion Matrix")

plt.tight_layout()
plt.savefig("reports/final_confusion_matrix.png", dpi=150)
plt.show()

# 1. Считаем количество ошибок каждого типа

print("\n===== ERROR TYPES =====")

error_types = (
    errors.groupby(
        ["true_sentiment", "predicted_sentiment"]
    )
    .size()
    .sort_values(ascending=False)
)

print(error_types)


# 2. Выводим примеры ошибок каждого типа

print("\n===== ERROR EXAMPLES =====")

for (true_class, predicted_class), group in errors.groupby(
    ["true_sentiment", "predicted_sentiment"],
    sort=False
):

    print(f"\n{true_class} -> {predicted_class}")
    print("Number of errors:", len(group))

    samples = group.sample(
        min(5, len(group)),
        random_state=42
    )

    for _, row in samples.iterrows():
        print("-", row["text"])

        errors.to_csv(
    "reports/error_analysis.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nError analysis saved successfully!")

# ==========================================
# DAY 5 — ERROR ANALYSIS SUMMARY
# ==========================================

error_summary = {
    "total_samples": len(error_df),
    "correct_predictions": len(error_df) - len(errors),
    "total_errors": len(errors),
    "error_rate": round(len(errors) / len(error_df), 4),
    "errors_by_true_class": errors["true_sentiment"].value_counts().to_dict(),
    "errors_by_predicted_class": errors["predicted_sentiment"].value_counts().to_dict(),
    "error_types": {
        f"{true} -> {pred}": int(count)
        for (true, pred), count in error_types.items()
    }
}

with open("reports/error_summary.json", "w", encoding="utf-8") as file:
    json.dump(error_summary, file, indent=4, ensure_ascii=False)

X = df_clean["processed_text"]
y = df_clean["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

tfidf_final = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_final = tfidf_final.fit_transform(X_train)
X_test_final = tfidf_final.transform(X_test)

final_model = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=42,
    max_iter=2000
)

final_model.fit(X_train_final, y_train)

os.makedirs("models", exist_ok=True)

joblib.dump(
    final_model,
    "models/linear_svc_model.pkl"
)

joblib.dump(
    tfidf_final,
    "models/tfidf_vectorizer.pkl"
)

print("Model and vectorizer saved successfully!")