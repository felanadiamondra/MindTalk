
import nltk
import json
import random
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import CountVectorizer


""" download nltk packages (first time only)

    nltk.download('punkt', force=True)
    nltk.download('stopwords', force=True)
    nltk.download('wordnet', force=True)
    nltk.download('vader_lexicon', force=True)
    nltk.download('averaged_perceptron_tagger', force=True)
    nltk.download('punkt_tab', force=True)

"""

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

""" Text preprocessing pipeline

"""

def preprocess_text(text):
    tokens = word_tokenize(text)
    filtered_tokens = [w.lower() for w in tokens if w.isalpha()
                       and w.lower() not in stop_words]

    lemmas = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmas)

# with open('positive_data.json', 'r') as f:
#     positive_data = json.load(f)

# with open('negative_data.json', 'r') as f:
#     negative_data = json.load(f)

# data = positive_data + negative_data

# Combines and shuffles data for randomized training and testing splits, preventing bias in model training.
# random.shuffle(data)

# print(f"Total samples loaded: {len(data)}")

# Visualize and count label distribution for data balance
# df = pd.DataFrame(data)
# print(df['label'].value_counts())


df = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')

df.head()

df['reviewText'] = df['reviewText'].apply(preprocess_text)
df

# print(df[['sentence', 'processed_text']].head())

# df

# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(df['processed_text'])
# y = df['label'].map({'positive': 1, 'negative': 0}).values

# print(f"Feature matrix shape: {X.shape}")