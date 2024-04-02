import joblib
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load trained model and vectorizer
rf_clf = joblib.load('random_forest_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Define preprocessing function
def preprocess_data(text):
    tokens = nlp(text)
    filtered_tokens = []
    for token in tokens:
        if not token.is_punct and not token.is_stop:
            filtered_tokens.append(token.lemma_)
    return " ".join(filtered_tokens)

# Function to predict sentiment
def predict_sentiment(text):
    preprocessed_text = preprocess_data(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = rf_clf.predict(vectorized_text)
    return prediction[0]

# Mapping dictionary for sentiment labels
sentiment_labels = {0: 'Irrelevant', 1: 'Negative', 2: 'Neutral', 3: 'Positive'}

# Streamlit app
def main():
    st.title('Sentiment Analysis Web App')
    st.write('Enter text for sentiment analysis below:')

    text_input = st.text_area('Text')

    if st.button('Analyze'):
        if text_input:
            sentiment_num = predict_sentiment(text_input)
            sentiment_label = sentiment_labels[sentiment_num]
            st.write(f'Sentiment: {sentiment_label}')
        else:
            st.warning('Please enter some text for analysis.')

if __name__ == '__main__':
    main()
