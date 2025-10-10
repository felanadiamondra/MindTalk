import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK data if not already done
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Streamlit app
st.set_page_config(page_title="Feelings Detector", page_icon="💬")
st.title("💬 Sentiment Analyzer")
st.write("Tell me how you feel, and I’ll give you some friendly feedback 🌈")

# User input
user_input = st.text_area("What's on your mind today?")

if st.button("Analyze"):
    if user_input.strip():
        # Get sentiment scores
        sentiment = sia.polarity_scores(user_input)
        compound = sentiment['compound']

        # Determine sentiment category
        if compound >= 0.05:
            category = "😊 Positive"
            feedback = "That’s great! Keep up the good vibes ✨"
        elif compound <= -0.05:
            category = "😞 Negative"
            feedback = "Sorry to hear that. Take a deep breath, everything will be okay 💛"
        else:
            category = "😐 Neutral"
            feedback = "Seems you’re feeling okay. Maybe a little music or a walk would brighten your day 🎶"

        # Display results
        st.subheader("Your Sentiment Result:")
        st.write(f"**Sentiment:** {category}")
        st.progress((compound + 1) / 2)
        st.write(f"**Feedback:** {feedback}")

    else:
        st.warning("Please enter a sentence before clicking Analyze.")
