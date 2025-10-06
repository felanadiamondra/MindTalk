
import nltk
import json
import random
import pandas as pd

""" download nltk packages (first time only)

    nltk.download('punkt', force=True)
    nltk.download('stopwords', force=True)
    nltk.download('wordnet', force=True)
    nltk.download('vader_lexicon', force=True)
    nltk.download('averaged_perceptron_tagger', force=True)
    nltk.download('punkt_tab', force=True)

"""

with open('positive_data.json', 'r') as f:
    positive_data = json.load(f)

with open('negative_data.json', 'r') as f:
    negative_data = json.load(f)

data = positive_data + negative_data

# Combines and shuffles data for randomized training and testing splits, preventing bias in model training.
random.shuffle(data)

# print(f"Total samples loaded: {len(data)}")

# Visualize and count label distribution for data balance
df = pd.DataFrame(data)
print(df['label'].value_counts())

df.head()