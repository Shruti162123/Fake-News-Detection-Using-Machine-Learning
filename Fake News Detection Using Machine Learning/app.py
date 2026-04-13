import argparse
import re
import string

import pandas as pd
from nltk.corpus import stopwords
from nltk import download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from visualizations import generate_research_graphs


# Download stopwords once (safe to call every run).
download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))


def preprocess_text(text):
    """
    Basic text cleaning:
    1) lowercase
    2) remove punctuation
    3) remove stopwords
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return " ".join(words)


def load_and_prepare_data(true_path="True.csv", fake_path="Fake.csv"):
    """Load True/Fake datasets, add labels, combine, and preprocess text."""
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)

    # Add labels: 1 = Real, 0 = Fake
    true_df["label"] = 1
    fake_df["label"] = 0

    # Combine both datasets into one DataFrame
    df = pd.concat([true_df, fake_df], ignore_index=True)

    # Combine title + text into one column for training
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["content"] = df["title"] + " " + df["text"]

    # Clean content
    df["content"] = df["content"].apply(preprocess_text)

    return df


def train_model(df):
    """Vectorize text, split data, train Logistic Regression, and evaluate."""
    X = df["content"]
    y = df["label"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\nModel Evaluation")
    print("-" * 30)
    print(f"Accuracy Score: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Save research-paper graphs in the "graphs" folder.
    generate_research_graphs(
        df=df,
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        model=model,
        vectorizer=vectorizer,
        output_dir="graphs",
    )

    return model, vectorizer


def predict_news(news_text, model, vectorizer):
    """Predict if given news text is Real or Fake."""
    cleaned_text = preprocess_text(news_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    return "Real News" if prediction == 1 else "Fake News"


def run_cli_prediction(model, vectorizer):
    """Simple terminal-based prediction loop."""
    print("\nEnter a news headline or article text to check.")
    print("Type 'exit' to stop.\n")

    while True:
        user_input = input("News text: ").strip()
        if user_input.lower() == "exit":
            print("Exiting prediction system.")
            break
        if not user_input:
            print("Please enter some text.")
            continue

        result = predict_news(user_input, model, vectorizer)
        print(f"Prediction: {result}\n")


def run_streamlit_app(model, vectorizer):
    """
    Optional Streamlit UI.
    Run with:
    streamlit run app.py -- --streamlit
    """
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit is not installed. Install it with: pip install streamlit")
        return

    st.title("Fake News Detection")
    st.write("Enter news text and click the button to classify it.")

    news_input = st.text_area("News Text", height=200)
    if st.button("Check News"):
        if not news_input.strip():
            st.warning("Please enter news text.")
        else:
            result = predict_news(news_input, model, vectorizer)
            st.success(f"Prediction: {result}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Run optional Streamlit UI (use with streamlit run).",
    )
    args = parser.parse_args()

    print("Loading data and training model...")
    data = load_and_prepare_data("True.csv", "Fake.csv")
    model, vectorizer = train_model(data)

    if args.streamlit:
        run_streamlit_app(model, vectorizer)
    else:
        run_cli_prediction(model, vectorizer)


if __name__ == "__main__":
    main()
