import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime
import pandas as pd
import random
import unicodedata
import time

# --- Configuration ---
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
st.set_page_config(page_title="MindTalk", page_icon="💬", layout="wide")

# --- Pastel theme ---
PRIMARY_COLOR = "#9B6BA7"
SECONDARY_COLOR = "#4D87A6"
BG_LIGHT = "#FAF7F2"
BG_GREY = "#63605f"

# --- Variables ---
category = ""
mood_key = ""
plan = ""
activities = []

# --- Feedback lists ---
positive_responses = [
    "🌞 Great! It’s wonderful to see you feeling good ✨",
    "💪 Keep that good mood going — it’s contagious! 😄",
    "🌸 So happy to read this! Keep spreading your good vibes 🌈",
    "😄 Awesome! Your smile lights up your day ☀️"
]

negative_responses = [
    "💛 I understand — tough days happen. Take care of yourself 💫",
    "🌧 Clouds pass… tomorrow will be kinder 🌤",
    "🤗 Stay strong, one small step at a time 💕",
    "💬 You did well opening up. Letting it out is already progress 🌱"
]

neutral_responses = [
    "🍃 A calm and balanced day is already valuable.",
    "🕊 Sometimes not feeling anything special is a sign of peace.",
    "🌼 A neutral day might be the calm before something beautiful 🌈",
    "☕ A quiet moment is perfect for recentring."
]

# --- Plans and activities ---
action_plans = {
    "positive": {
        "plan": [
            "Keep nurturing your positive energy by sharing a moment with someone you love 💕",
            "Write down three things that made you happy today ✨",
            "Do something you love: music, dancing, cooking 🥗"
        ],
        "activities": [
            "📖 Read an inspiring book",
            "🎶 Listen to your favorite playlist",
            "🚶 Take a 10-minute walk outside",
            "💌 Send a kind message to someone"
        ]
    },
    "negative": {
        "plan": [
            "Take a moment to breathe deeply and refocus 🧘",
            "Try to put your feelings into words 💭",
            "Remember: emotions are temporary 💛"
        ],
        "activities": [
            "🎧 Listen to something calm",
            "✍️ Write in a journal",
            "🌿 Take a peaceful walk",
            "📞 Call a trusted friend"
        ]
    },
    "neutral": {
        "plan": [
            "Use this stability to recharge 🍃",
            "Try a new activity to boost your creativity 🎨",
            "Plan something pleasant for tomorrow 🌅"
        ],
        "activities": [
            "🧩 Do a puzzle or logic game",
            "📺 Watch an inspiring movie",
            "☕ Enjoy a quiet moment without your phone",
            "🧘 Try a short guided meditation"
        ]
    }
}

emotion_keywords = {
    "fatigue": ["fatigue", "tired", "exhausted", "drained", "sleepy"],
    "joy": ["happy", "joyful", "satisfied", "glad", "excited"],
    "anger": ["angry", "mad", "furious", "upset", "annoyed"],
    "stress": ["stress", "stressed", "anxious", "pressure", "tense"],
    "sadness": ["sad", "down", "depressed", "blue", "melancholic"]
}

def normalize_text(text):
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    return text

# --- Greeting ---
hour = datetime.now().hour
if hour < 12:
    greeting = "☀️ Good morning! Ready for a new day?"
elif hour < 18:
    greeting = "🌤 Good afternoon! How are you feeling?"
else:
    greeting = "🌙 Good evening! Want to reflect on your day?"

st.markdown(f"<h2 style='color:{PRIMARY_COLOR}; text-align:center'>{greeting}</h2>", unsafe_allow_html=True)
st.markdown(f"<h1 style='text-align:center; color:{SECONDARY_COLOR}'>💬 MindTalk</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555'>Take a moment for yourself and share how you feel 🌈</p>", unsafe_allow_html=True)

# --- Layout (2 main columns) ---
main_col, side_col = st.columns([2, 1])

# LEFT COLUMN 
with main_col:
    st.markdown("---")
    st.subheader("📝 Your mood today")
    user_input = st.text_area("Express yourself freely:", placeholder="Write your thoughts here...", height=120)

    if "history" not in st.session_state:
        st.session_state.history = []

    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    with col_btn1:
        analyze_button = st.button("💬 Analyze my mood")
    with col_btn2:
        clear_button = st.button("🗑 Clear history")
    with col_btn3:
        if st.session_state.history:
            st.download_button(
                label="📥 Download",
                data=pd.DataFrame(st.session_state.history).to_csv(index=False),
                file_name="mindtalk_history.csv",
                mime="text/csv"
            )
        else:
            st.button("📥 Download", disabled=True)

    if clear_button:
        st.session_state.history = []
        st.success("History cleared successfully! ✨")

    # --- Main analysis ---
    if analyze_button:
        if user_input.strip():
            sentiment = sia.polarity_scores(user_input)
            compound = sentiment["compound"]
            current_time = datetime.now().strftime("%d/%m/%Y %H:%M")

            if compound >= 0.05:
                category = "😊 Positive"
                feedback = random.choice(positive_responses)
                st.success(feedback)
                mood_key = "positive"
            elif compound <= -0.05:
                category = "😞 Negative"
                feedback = random.choice(negative_responses)
                st.error(feedback)
                mood_key = "negative"
            else:
                category = "😐 Neutral"
                feedback = random.choice(neutral_responses)
                st.info(feedback)
                mood_key = "neutral"

            st.progress((compound + 1) / 2)
            st.caption(f"**Overall sentiment: {category}**")

            text = normalize_text(user_input.lower())
            for key, words in emotion_keywords.items():
                if any(w in text for w in words):
                    if key == "fatigue":
                        st.write("💤 You seem tired. Take a moment for yourself.")
                    elif key == "joy":
                        st.write("🌞 So glad to see you happy! Keep smiling 💕")
                    elif key == "anger":
                        st.write("😤 Take a deep breath and step back. It will get better.")
                    elif key == "stress":
                        st.write("🧘 Inhale… exhale… You got this 🌿")
                    elif key == "sadness":
                        st.write("🌱 Try writing down something that made you smile today.")

            # Generate plan and activities
            plan = random.choice(action_plans[mood_key]["plan"])
            activities = random.sample(action_plans[mood_key]["activities"], 2)

            st.session_state.history.append({
                "text": user_input,
                "category": category,
                "feedback": feedback,
                "time": current_time,
                "compound": compound
            })
        else:
            st.warning("✏️ Please write something before analyzing your mood!")

    # --- Summary and History ---
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📊 Mood Statistics")
        pos = sum(1 for h in st.session_state.history if "Positive" in h["category"])
        neg = sum(1 for h in st.session_state.history if "Negative" in h["category"])
        neu = sum(1 for h in st.session_state.history if "Neutral" in h["category"])
        st.write(f"**😊 Positive: {pos} | 😞 Negative: {neg} | 😐 Neutral: {neu}**")

        st.markdown("---")
        st.subheader("🕓 Recent History")
        for entry in reversed(st.session_state.history[-5:]):
            st.markdown(
                f"""
                <div style="background-color:{BG_GREY}; padding:10px; border-radius:10px; margin-bottom:5px;">
                    <b>🗣️ {entry['text']}</b><br>
                    → {entry['category']}<br>
                    💡 <i>{entry['feedback']}</i><br>
                    🕒 {entry['time']}
                </div>
                """,
                unsafe_allow_html=True
            )

#  RIGHT COLUMN 
with side_col:
    st.markdown("---")
    st.subheader("Action Plan & Activities")
    if plan or activities:
        with st.spinner("Wait a moment 🧭", show_time=False):
            time.sleep(1)

        if plan:
            st.markdown(f"<div style='background-color:{BG_GREY}; padding:10px; border-radius:10px;'>💡 {plan}</div>", unsafe_allow_html=True)
        if activities:
            st.markdown("### 🎯 Suggested activities")
            for activity in activities:
                st.markdown(f"- {activity}")
    else:
        st.info("Analyze your mood to discover adapted activities 💬")
