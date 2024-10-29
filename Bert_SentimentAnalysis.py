#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import tarfile
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


current_folder = os.getcwd()
dataset = tf.keras.utils.get_file(fname ="aclImdb.tar.gz", origin ="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                                  cache_dir= current_folder, extract = True)


# In[3]:


dataset_path = os.path.dirname(dataset)
os.listdir(dataset_path)


# In[4]:


dataset_dir = os.path.join(dataset_path, 'aclImdb')
os.listdir(dataset_dir)


# In[5]:


train_dir = os.path.join(dataset_dir,'train')
os.listdir(train_dir)


# In[6]:


def load_dataset(directory):
    data = {"sentence": [], "sentiment": []}
    for file_name in os.listdir(directory):
        print(file_name)
        if file_name == 'pos':
            positive_dir = os.path.join(directory, file_name)
            for text_file in os.listdir(positive_dir):
                text = os.path.join(positive_dir, text_file)
                with open(text, "r", encoding="utf-8") as f:
                    data["sentence"].append(f.read())
                    data["sentiment"].append(1)
        elif file_name == 'neg':
            negative_dir = os.path.join(directory, file_name)
            for text_file in os.listdir(negative_dir):
                text = os.path.join(negative_dir, text_file)
                with open(text, "r", encoding="utf-8") as f:
                    data["sentence"].append(f.read())
                    data["sentiment"].append(0)
    return pd.DataFrame.from_dict(data)


# In[7]:


train_df = load_dataset(train_dir)
print(train_df.head())


# In[8]:


test_dir = os.path.join(dataset_dir,'test')
test_df = load_dataset(test_dir)
print(test_df.head())


# In[59]:


print(test_df[:1])


# In[11]:


sentiment_counts = train_df['sentiment'].value_counts()

fig = px.bar(
    x={0: 'Negative', 1: 'Positive'},
    y=sentiment_counts.values,
    color=sentiment_counts.index,
    color_discrete_sequence=px.colors.qualitative.Dark24,
    title='<b>Sentiments Counts'
)

fig.update_layout(
    title='Sentiments Counts',
    xaxis_title='Sentiment',
    yaxis_title='Counts'
)

fig.show()


# In[9]:


def text_cleaning(text):
    soup = BeautifulSoup(text, "html.parser")
    text = re.sub(r'\[[^]]*\]', '', soup.get_text())
    pattern = r"[^a-zA-Z0-9\s,']"
    text = re.sub(pattern, '', text)
    return text


# In[10]:


train_df['Cleaned_sentence'] = train_df['sentence'].apply(text_cleaning).tolist()
test_df['Cleaned_sentence'] = test_df['sentence'].apply(text_cleaning)


# In[14]:


def generate_wordcloud(text, Title):
    all_text = " ".join(text)
    wordcloud = WordCloud(width=600, height=600, stopwords=set(STOPWORDS), background_color='white').generate(all_text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(Title)
    plt.show()


# In[15]:


positive = train_df[train_df['sentiment']==1]['Cleaned_sentence'].tolist()
generate_wordcloud(positive,'Positive Review')


# In[16]:


negative = train_df[train_df['sentiment']==0]['Cleaned_sentence'].tolist()
generate_wordcloud(negative,'Negative Review')


# In[11]:


Reviews = train_df['Cleaned_sentence']
Target = train_df['sentiment']
test_reviews = test_df['Cleaned_sentence']
test_targets = test_df['sentiment']


# In[12]:


x_val, x_test, y_val, y_test = train_test_split(test_reviews, test_targets, test_size=0.5, stratify = test_targets)


# In[13]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
max_len= 128
X_train_encoded = tokenizer.batch_encode_plus(Reviews.tolist(), padding=True, truncation=True,
                                              max_length = max_len, return_tensors='tf')
X_val_encoded = tokenizer.batch_encode_plus(x_val.tolist(), padding=True, truncation=True,
                                            max_length = max_len, return_tensors='tf')
X_test_encoded = tokenizer.batch_encode_plus(x_test.tolist(), padding=True, truncation=True,
                                             max_length = max_len, return_tensors='tf')


# In[53]:


model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])


# In[ ]:


dhistory = model.fit(
    [X_train_encoded['input_ids'], X_train_encoded['token_type_ids'], X_train_encoded['attention_mask']],
    Target,
    validation_data=(
        [X_val_encoded['input_ids'], X_val_encoded['token_type_ids'], X_val_encoded['attention_mask']],
        y_val
    ),
    batch_size=32,
    epochs=3
)


# In[ ]:


test_loss, test_accuracy = model.evaluate([X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']],y_test)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')


# In[ ]:


path = "/save-to-folder"


# In[ ]:


tokenizer.save_pretrained(path +"/tokenizer")
model.save_pretrained(path + "/model")
model.save_weights(path +"/weights")


# In[14]:


bert_tokenizer = BertTokenizer.from_pretrained("ModelHistory/Tokenizer")
bert_model = TFBertForSequenceClassification.from_pretrained("ModelHistory/Model")


# In[15]:


pred = bert_model.predict([X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']])
logits = pred.logits
pred_labels = tf.argmax(logits, axis=1)
pred_labels = pred_labels.numpy()
label = {
1: 'positive',
0: 'Negative'
}
pred_labels = [label[i] for i in pred_labels]
Actual = [label[i] for i in y_test]
print('Predicted Label :', pred_labels[:10])
print('Actual Label :', Actual[:10])


# In[16]:


print("Classification Report: \n", classification_report(Actual, pred_labels))


# In[17]:


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


# In[21]:


Review ='''This is a journey not a destination film. Used to be a lot of films like this. Today, not so much. That said, for its genre, it is top
of class. Burke has an everyman look and attitude which Jenkins uses to advantage. The characters, all superb, are people you want to meet once in
your life, but only once. Recommended.'''
Get_sentiment(Review)


# In[ ]:




