import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime
import pandas as pd
import random
import unicodedata
import time

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="MindTalk",
    page_icon="💬",
    layout="wide"
)

# --------------------------------------------------
# Safe NLTK initialization
# --------------------------------------------------
@st.cache_resource
def load_sentiment_analyzer():
    try:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception as e:
        st.error("⚠️ Sentiment engine could not be loaded.")
        st.stop()

sia = load_sentiment_analyzer()

# --------------------------------------------------
# Theme
# --------------------------------------------------
PRIMARY_COLOR = "#9B6BA7"
SECONDARY_COLOR = "#4D87A6"
BG_GREY = "#63605f"

# --------------------------------------------------
# Session state initialization
# --------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "plan" not in st.session_state:
    st.session_state.plan = None

if "activities" not in st.session_state:
    st.session_state.activities = []

# --------------------------------------------------
# Text utils
# --------------------------------------------------
def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFD", text).encode(
        "ascii", "ignore"
    ).decode("utf-8")

# --------------------------------------------------
# Feedback
# --------------------------------------------------
positive_responses = [
    "🌞 Great! It’s wonderful to see you feeling good ✨",
    "💪 Keep that good mood going — it’s contagious! 😄",
    "🌸 So happy to read this! Keep spreading your good vibes 🌈",
    "😄 Awesome! Your smile lights up your day ☀️",
]

negative_responses = [
    "💛 I understand — tough days happen. Take care of yourself 💫",
    "🌧 Clouds pass… tomorrow will be kinder 🌤",
    "🤗 Stay strong, one small step at a time 💕",
    "💬 Opening up is already progress 🌱",
]

neutral_responses = [
    "🍃 A calm and balanced day is already valuable.",
    "🕊 Sometimes neutral means peaceful.",
    "🌼 A quiet day can be restorative.",
    "☕ A pause is sometimes exactly what we need.",
]

# --------------------------------------------------
# Plans & activities
# --------------------------------------------------
action_plans = {
    "positive": {
        "plan": [
            "Share your energy with someone you care about 💕",
            "Write down three good moments from today ✨",
            "Do something you truly enjoy 🎶",
        ],
        "activities": [
            "📖 Read an inspiring book",
            "🎶 Listen to your favorite playlist",
            "🚶 Take a short walk",
            "💌 Send a kind message",
        ],
    },
    "negative": {
        "plan": [
            "Pause and breathe slowly 🧘",
            "Write your thoughts down 💭",
            "Remember: emotions pass 💛",
        ],
        "activities": [
            "🎧 Listen to calm sounds",
            "✍️ Journal for 5 minutes",
            "🌿 Step outside briefly",
            "📞 Reach out to someone you trust",
        ],
    },
    "neutral": {
        "plan": [
            "Use this calm to recharge 🍃",
            "Try something creative 🎨",
            "Plan something pleasant for tomorrow 🌅",
        ],
        "activities": [
            "🧩 Do a puzzle",
            "📺 Watch something uplifting",
            "☕ Enjoy a quiet moment",
            "🧘 Short meditation",
        ],
    },
}

emotion_keywords = {
    "fatigue": ["fatigue", "tired", "exhausted", "drained", "sleepy"],
    "joy": ["happy", "joyful", "glad", "excited"],
    "anger": ["angry", "mad", "furious", "annoyed"],
    "stress": ["stress", "stressed", "anxious", "tense"],
    "sadness": ["sad", "down", "blue", "melancholic"],
}

# --------------------------------------------------
# Greeting
# --------------------------------------------------
hour = datetime.now().hour
greeting = (
    "☀️ Good morning!"
    if hour < 12
    else "🌤 Good afternoon!"
    if hour < 18
    else "🌙 Good evening!"
)

st.markdown(
    f"<h2 style='color:{PRIMARY_COLOR}; text-align:center'>{greeting}</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<h1 style='text-align:center; color:{SECONDARY_COLOR}'>💬 MindTalk</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#555'>Take a moment for yourself 🌈</p>",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Layout
# --------------------------------------------------
main_col, side_col = st.columns([2, 1])

# ==================================================
# LEFT COLUMN
# ==================================================
with main_col:
    st.subheader("📝 Your mood today")
    user_input = st.text_area(
        "Express yourself freely:",
        placeholder="Write your thoughts here...",
        height=120,
        max_chars=1000,
    )

    col1, col2, col3 = st.columns([2, 1, 1])

    analyze = col1.button("💬 Analyze my mood")
    clear = col2.button("🗑 Clear history")

    if st.session_state.history:
        col3.download_button(
            "📥 Download",
            pd.DataFrame(st.session_state.history).to_csv(index=False),
            "mindtalk_history.csv",
            "text/csv",
        )
    else:
        col3.button("📥 Download", disabled=True)

    if clear:
        st.session_state.history.clear()
        st.session_state.plan = None
        st.session_state.activities = []
        st.success("History cleared ✨")

    # ---------------- Analysis ----------------
    if analyze:
        if not user_input.strip():
            st.warning("✏️ Please write something first.")
        else:
            try:
                scores = sia.polarity_scores(user_input)
                compound = scores["compound"]
                timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")

                if compound >= 0.05:
                    mood_key = "positive"
                    category = "😊 Positive"
                    feedback = random.choice(positive_responses)
                    st.success(feedback)
                elif compound <= -0.05:
                    mood_key = "negative"
                    category = "😞 Negative"
                    feedback = random.choice(negative_responses)
                    st.error(feedback)
                else:
                    mood_key = "neutral"
                    category = "😐 Neutral"
                    feedback = random.choice(neutral_responses)
                    st.info(feedback)

                st.progress((compound + 1) / 2)
                st.caption(f"**Overall sentiment: {category}**")

                text = normalize_text(user_input.lower())
                for emotion, words in emotion_keywords.items():
                    if any(w in text for w in words):
                        st.write(f"💬 Detected emotion: **{emotion.capitalize()}**")
                        break

                st.session_state.plan = random.choice(
                    action_plans[mood_key]["plan"]
                )
                st.session_state.activities = random.sample(
                    action_plans[mood_key]["activities"], 2
                )

                st.session_state.history.append(
                    {
                        "text": user_input,
                        "category": category,
                        "feedback": feedback,
                        "compound": compound,
                        "time": timestamp,
                    }
                )

            except Exception:
                st.error("⚠️ Something went wrong while analyzing your mood.")

    # --------------------------------------------------
    # History & Statistics (RESTORED)
    # --------------------------------------------------
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📊 Mood Statistics")

        pos = sum("Positive" in h["category"] for h in st.session_state.history)
        neg = sum("Negative" in h["category"] for h in st.session_state.history)
        neu = sum("Neutral" in h["category"] for h in st.session_state.history)

        st.write(f"**😊 Positive: {pos} | 😞 Negative: {neg} | 😐 Neutral: {neu}**")

        st.markdown("---")
        st.subheader("🕓 Recent History")

        for entry in reversed(st.session_state.history[-5:]):
            st.markdown(
                f"""
                <div style="background-color:{BG_GREY};
                            padding:10px;
                            border-radius:10px;
                            margin-bottom:8px;">
                    <b>🗣️ {entry['text']}</b><br>
                    → {entry['category']}<br>
                    💡 <i>{entry['feedback']}</i><br>
                    🕒 {entry['time']}
                </div>
                """,
                unsafe_allow_html=True
            )


# ==================================================
# RIGHT COLUMN
# ==================================================
with side_col:
    st.subheader("🧭 Plan & Activities")

    if st.session_state.plan:
        with st.spinner("Preparing suggestions…"):
            time.sleep(0.6)

        st.markdown(
            f"<div style='background-color:{BG_GREY}; padding:10px; border-radius:10px;'>💡 {st.session_state.plan}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### 🎯 Suggested activities")
        for act in st.session_state.activities:
            st.markdown(f"- {act}")
    else:
        st.info("Analyze your mood to receive guidance 💬")
