import streamlit as st
import os
import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import re
from bs4 import BeautifulSoup

# Load saved model and tokenizer
path = "C:/xampp/htdocs/python/try/working/"
bert_tokenizer = BertTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
bert_model = TFBertForSequenceClassification.from_pretrained(os.path.join(path, "model"))

label = {
    1: 'positive',
    0: 'negative'
}

def Get_sentiment(Review, Tokenizer=bert_tokenizer, Model=bert_model):
    if not isinstance(Review, list):
        Review = [Review]
    Input_ids, Token_type_ids, Attention_mask = Tokenizer.batch_encode_plus(
        Review, padding=True, truncation=True, max_length=128, return_tensors='tf'
    ).values()
    prediction = Model.predict([Input_ids, Token_type_ids, Attention_mask])
    pred_labels = tf.argmax(prediction.logits, axis=1)
    pred_labels = [label[i] for i in pred_labels.numpy().tolist()]
    return pred_labels

# Function to perform text cleaning
def text_cleaning(text):
    soup = BeautifulSoup(text, "html.parser")
    text = re.sub(r'\[[^]]*\]', '', soup.get_text())
    pattern = r"[^a-zA-Z0-9\s,']"
    text = re.sub(pattern, '', text)
    return text

# Function to perform sentiment analysis
def perform_sentiment_analysis(text):
    cleaned_text = text_cleaning(text)
    pred_label = Get_sentiment(cleaned_text, Tokenizer=bert_tokenizer, Model=bert_model)[0]  # Get the first prediction
    return pred_label

# Sidebar for file upload
st.sidebar.title("Upload File")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Main content area
st.title("Sentiment Analysis with BERT")

if uploaded_file is not None:
    # Load test dataset
    df = pd.read_csv(uploaded_file)

    # Initialize counters for evaluation metrics
    total_sentences = len(df)
    correct_predictions = 0
    positive_count = 0
    negative_count = 0

    # Display metrics at the top of the page
    st.subheader("Evaluation Metrics")

    # Calculate accuracy and sentiment counts
    for index, row in df.iterrows():
        prediction = perform_sentiment_analysis(row["Cleaned_sentence"])
        if prediction == row["sentiment"]:
            correct_predictions += 1
        if prediction == "positive":
            positive_count += 1
        else:
            negative_count += 1
    
    accuracy = correct_predictions / total_sentences

    # Graphics showing the number of positive and negative predictions
    st.write(f"Number of Positive Predictions: {positive_count}")
    st.write(f"Number of Negative Predictions: {negative_count}")
    
    # Display results for correct predictions
    st.subheader("Correct Predictions")
    for index, row in df.iterrows():
        prediction = perform_sentiment_analysis(row["Cleaned_sentence"])
        st.write(f"### Row {index + 1}:")
        st.write("Original Text:", row["Cleaned_sentence"])
        actual_sentiment_label = row["sentiment"]
        actual_sentiment_str = "positive" if actual_sentiment_label == 1 else "negative"
        st.write("Predicted Sentiment:", prediction)
        st.write("Actual Sentiment (Numeric):", actual_sentiment_label)
        st.write("---")


    # Display accuracy
    st.write(f"Total Sentences: {total_sentences}")
    st.write(f"Correct Predictions: {correct_predictions}")
    st.write(f"Accuracy: {accuracy:.2%}")  # Display accuracy as a percentage
