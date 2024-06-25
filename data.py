import pandas as pd
import random
import nltk
from nltk.corpus import movie_reviews

# Download the IMDb movie reviews dataset
nltk.download("movie_reviews")

# Get positive and negative movie reviews
positive_reviews = [(movie_reviews.raw(fileid), 'positive') for fileid in movie_reviews.fileids("pos")]
negative_reviews = [(movie_reviews.raw(fileid), 'negative') for fileid in movie_reviews.fileids("neg")]

# Shuffle the reviews
random.shuffle(positive_reviews)
random.shuffle(negative_reviews)

# Select a subset of 50 positive and 50 negative reviews

positive_reviews_subset = positive_reviews[:40]
negative_reviews_subset = negative_reviews[:60]

# Combine positive and negative reviews
combined_reviews = positive_reviews_subset + negative_reviews_subset

# Shuffle the combined reviews
random.shuffle(combined_reviews)

# Create DataFrame
df = pd.DataFrame(combined_reviews, columns=['Cleaned_sentence', 'Actual_sentiment'])

file_path = 'C:/xampp/htdocs/python/try/working/random_movie_reviews.csv'

# Save to CSV
df.to_csv(file_path, index=False)

print(f"CSV file saved successfully at {file_path}")
