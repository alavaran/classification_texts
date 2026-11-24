import re
import joblib

# Load trained artifacts
model = joblib.load("models/linear_svc_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

print("Model loaded:", type(model).__name__)
print("Vectorizer loaded:", type(vectorizer).__name__)

def preprocess_text(text: str) -> str:
    """
    Preprocess a financial news sentence.
    """
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# Load trained artifacts
model = joblib.load("models/linear_svc_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


LABEL_MAPPING = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

def predict_sentiment(text: str) -> dict:
    """
    Predict sentiment for a single financial news sentence.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    if not text.strip():
        raise ValueError("Input text cannot be empty.")

    processed_text = preprocess_text(text)

    if not processed_text:
        raise ValueError("Text is empty after preprocessing.")

    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]

    sentiment = LABEL_MAPPING[prediction]

    return {
        "text": text,
        "processed_text": processed_text,
        "sentiment": sentiment
    }

if __name__ == "__main__":

    test_cases = [
        "",
        "   ",
        "<p></p>",
        "THE COMPANY REPORTED STRONG GROWTH!",
        "The company reported a loss of 20 million euros.",
        "The company reported strong revenue growth.",
        "The company announced significant financial losses.",
        "The company reported its financial results for the first quarter.",
        "Operating costs decreased by 15 percent.",
        "The company expects a decline in sales next year.",
        "The acquisition will have an immediate positive impact on financial results."
    ]

    for text in test_cases:
        try:
            result = predict_sentiment(text)
            print(result)

        except (TypeError, ValueError) as error:
            print(f"Input error: {error}")

