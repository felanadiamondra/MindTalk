import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def sentiment_analysis(sentence):
    # Use the SentimentIntensityAnalyzer to compute the sentiment scores
    sentiment = SentimentIntensityAnalyzer().polarity_scores(sentence)

    # Categorize the sentiment as positive, negative, or neutral based on the compound score
    if sentiment['compound'] >= 0.05:
        sentiment_category = "Positive"
    elif sentiment['compound'] <= -0.05:
        sentiment_category = "Negative"
    else:
        sentiment_category = "Neutral"

    return sentiment, sentiment_category


# Test the sentiment analysis on some example sentences
sentence = "I love this youtube video! You Rock."
sentiment, sentiment_category = sentiment_analysis(sentence)
print("Sentence:", sentence)
print("Compound score:", sentiment['compound'])
print("Sentiment:", sentiment_category)

sentence = "I hate this youtube video! You're Terrible."
sentiment, sentiment_category = sentiment_analysis(sentence)
print("Sentence:", sentence)
print("Compound score:", sentiment['compound'])
print("Sentiment:", sentiment_category)

sentence = "I feel so-so about your youtube videos."
sentiment, sentiment_category = sentiment_analysis(sentence)
print("Sentence:", sentence)
print("Compound score:", sentiment['compound'])
print("Sentiment:", sentiment_category)

sentence = "I feel so-so about your boring youtube videos."
sentiment, sentiment_category = sentiment_analysis(sentence)
print("Sentence:", sentence)
print("Compound score:", sentiment['compound'])
print("Sentiment:", sentiment_category)