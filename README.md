# Sentiment Analysis using BERT

## Description
This project leverages BERT for sequence classification to perform sentiment analysis on IMDb movie reviews. The objective is to classify movie reviews as positive or negative based on their textual content, providing insights into user opinions.

## Prerequisites and Installation
To run this project, you’ll need the following libraries and tools:

- **Python 3.7+**
- **TensorFlow**
- **Hugging Face Transformers**
- **Streamlit**
- **Other Libraries**: BeautifulSoup, matplotlib, plotly, scikit-learn, nltk, wordcloud

You can install the necessary dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Information
This project uses the IMDb movie reviews dataset, which is automatically downloaded and preprocessed. The `data.py` script helps generate a random subset of movie reviews for training and testing.

## Code Structure
- **`Bert_SentimentAnalysis.py`**: Main script for training and evaluating the BERT model on sentiment analysis.
- **`app.py`**: Streamlit app for performing real-time sentiment analysis on user-provided text or an uploaded CSV file.
- **`data.py`**: Script to download and preprocess movie reviews into a CSV file.

## Usage Instructions

### Running Model Training and Evaluation
To train and evaluate the BERT model, run the following command in your terminal:

```bash
python Bert_SentimentAnalysis.py
```

### Launching the Streamlit App
To start the Streamlit app, which allows you to perform sentiment analysis on custom inputs, run:

```bash
streamlit run app.py
```

With the app, you can upload a CSV file or input text to view the sentiment predictions.

## Example Outputs
Sample visualizations include:

- **Word Clouds**: Positive and negative word clouds generated from the reviews.
- **Sentiment Distribution**: Bar chart showing the distribution of positive and negative reviews in the dataset.

![image](https://github.com/user-attachments/assets/2593716d-6003-4c33-b02a-c3dadf2ffa10)
 
![image](https://github.com/user-attachments/assets/bdeb48c8-53a8-4e0f-b435-e261334ab25e)



Results and Model Performance
The BERT model was evaluated on the IMDb test dataset, with the following performance metrics:

Accuracy: The model achieved an overall accuracy of 87% in classifying positive and negative sentiments.
Classification Report: Precision, recall, and F1-score for each sentiment class are detailed below.
Classification Report:

                   precision    recall  f1-score   support

        Negative       0.88      0.85      0.87     6250
        Positive       0.86      0.88      0.87     6250

        accuracy                           0.87     12500
       macro avg       0.87      0.87      0.87     12500
    weighted avg       0.87      0.87      0.87     12500

### Interpretation:

Precision: The model correctly identifies 88% of negative and 86% of positive reviews out of all reviews predicted for each class.

Recall: The model successfully retrieves 85% of negative and 88% of positive reviews out of all true instances of each class.

F1-Score: Both positive and negative classes have a balanced F1-score of 0.87, indicating effective sentiment classification.

## Future Improvements
Possible enhancements for the project include:

Experimenting with Other Transformer Models: Trying models like RoBERTa or DistilBERT for potentially improved performance.

Dataset Expansion: Testing the model with additional datasets to make it more robust.

Prediction Optimization: Improving the model's inference speed for quicker real-time predictions.

Hyperparameter Tuning: Adjusting parameters like learning rate, batch size, and the number of epochs to improve model accuracy.

References
IMDb Dataset Source
Hugging Face Transformers Library
TensorFlow

## License
This project is open-source and available under the [MIT License](LICENSE).

