import random
import unicodedata
from datetime import datetime

import nltk
import pandas as pd
import plotly.express as px
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="MindTalk",
    page_icon="💬",
    layout="wide",
)

# --------------------------------------------------
# Theme
# --------------------------------------------------
PRIMARY = "#2F4B6E"      # slate blue
ACCENT = "#6B5B95"       # muted violet
TEXT = "#1F2933"
MUTED = "#6B7785"
BORDER = "#E3E8EF"
SURFACE = "#F7F9FC"

MOODS = {
    "positive": {"label": "Positive", "color": "#2E7D5B", "icon": ":material/sentiment_satisfied:"},
    "negative": {"label": "Negative", "color": "#B4474B", "icon": ":material/sentiment_dissatisfied:"},
    "neutral": {"label": "Neutral", "color": "#5B6B7F", "icon": ":material/sentiment_neutral:"},
}

st.markdown(
    f"""
    <style>
        html, body, [class*="css"] {{ color: {TEXT}; }}
        .block-container {{ padding-top: 2rem; max-width: 1200px; }}
        
        /* Purple Header Styling */
        .mt-header {{
            background: linear-gradient(135deg, {ACCENT} 0%, {PRIMARY} 100%);
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            color: #FFFFFF !important;
            box-shadow: 0 4px 15px rgba(107, 91, 149, 0.25);
        }}
        .mt-title {{
            font-size: 2rem; font-weight: 700; color: #FFFFFF !important;
            margin: 0; letter-spacing: -0.01em; display: flex; align-items: center; gap: 0.5rem;
        }}
        .mt-sub {{ color: rgba(255, 255, 255, 0.85) !important; font-size: 0.95rem; margin-top: 0.25rem; }}
        .mt-greeting {{
            color: rgba(255, 255, 255, 0.95) !important; font-size: 0.8rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.08em;
        }}
        
        .mt-entry {{
            background: {SURFACE};
            border: 1px solid {BORDER};
            border-left-width: 4px;
            border-radius: 8px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.6rem;
        }}
        .mt-entry .meta {{ color: {MUTED}; font-size: 0.8rem; }}
        .mt-badge {{
            display: inline-block; font-size: 0.75rem; font-weight: 600;
            padding: 0.1rem 0.55rem; border-radius: 999px; color: #fff;
        }}
        .mt-plan {{
            background: {SURFACE}; border: 1px solid {BORDER};
            border-left: 4px solid {ACCENT};
            border-radius: 8px; padding: 0.9rem 1rem; color: {TEXT};
        }}
        .mt-note {{ color: {MUTED}; font-size: 0.78rem; margin-top: 1.5rem; }}
        div.stButton > button, div.stDownloadButton > button {{
            border-radius: 6px; font-weight: 500;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Safe NLTK initialization
# --------------------------------------------------
@st.cache_resource
def load_sentiment_analyzer():
    try:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


sia = load_sentiment_analyzer()
if sia is None:
    st.error("The sentiment engine could not be loaded.", icon=":material/error:")
    st.stop()

# --------------------------------------------------
# Session state
# --------------------------------------------------
st.session_state.setdefault("history", [])
st.session_state.setdefault("plan", None)
st.session_state.setdefault("activities", [])
st.session_state.setdefault("user_input_val", "")


# --------------------------------------------------
# Utils & Hybrid Sentiment Analysis (FR + EN + Negations)
# --------------------------------------------------
def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")


FRENCH_LEXICON = {
    "positive": [
        "heureux", "heureuse", "joie", "content", "contente", "super", "genial", "génial",
        "magnifique", "merveilleux", "bien", "bonheur", "sourire", "amour", "aimer", "paix",
        "calme", "serein", "optimiste", "motivé", "motivée", "top", "excellent", "parfait"
    ],
    "negative": [
        "triste", "tristesse", "deprime", "déprimé", "déprimée", "seul", "seule", "solitude",
        "mal", "mauvais", "peur", "angoisse", "anxiété", "anxieux", "anxieuse", "pleurer",
        "pleurs", "souffrance", "douloureux", "douleur", "horrible", "nul", "nulle", "desespoir",
        "désespoir", "vide", "abandonné", "abandonnée", "sombre", "deçu", "déçu", "déçue"
    ],
    "anger": [
        "colere", "colère", "enervé", "énervé", "énervée", "faché", "fâché", "fâchée", "furieux",
        "rage", "haine", "agacé", "agacée", "irrité", "irritée", "injuste"
    ],
    "stress": [
        "stress", "stresse", "stressé", "stressée", "anxieux", "anxieuse", "pression", "panique",
        "submergé", "submergée", "dépassé", "dépassée", "tendu", "tendue"
    ],
    "fatigue": [
        "fatigue", "fatigué", "fatiguée", "épuisé", "épuisée", "crevé", "crevée", "draine",
        "drainé", "drainée", "hs", "sommeil", "lassitude", "à bout"
    ]
}

NEGATIONS = ["pas", "non", "plus", "jamais", "aucun", "sans", "not", "no", "never"]
INTENSIFIERS = ["très", "tres", "trop", "tellement", "vraiment", "extrêmement", "very", "really"]


def analyze_hybrid_sentiment(text: str) -> float:
    normalized = normalize_text(text.lower())
    words = normalized.split()
    
    pos_count = 0.0
    neg_count = 0.0
    
    for i, word in enumerate(words):
        is_negated = i > 0 and words[i-1] in NEGATIONS
        multiplier = 1.5 if (i > 0 and words[i-1] in INTENSIFIERS) else 1.0

        if is_negated and word in ["mal", "mauvais"]:
            pos_count += 1.5 * multiplier
            continue
        elif is_negated and word in ["triste", "déprimé", "horrible", "bad", "sad"]:
            pos_count += 1.0 * multiplier
            continue
        elif is_negated and word in FRENCH_LEXICON["positive"]:
            neg_count += 1.0 * multiplier
            continue

        if word in FRENCH_LEXICON["positive"]:
            pos_count += 1.0 * multiplier
        elif word in FRENCH_LEXICON["negative"]:
            neg_count += 1.0 * multiplier

    fr_total = pos_count + neg_count
    if fr_total > 0:
        return (pos_count - neg_count) / fr_total
    
    return sia.polarity_scores(text)["compound"]


# --------------------------------------------------
# Content
# --------------------------------------------------
feedback_messages = {
    "positive": [
        "Your message reflects a positive state of mind. Keep building on it.",
        "A good dynamic today. Take a moment to recognise what is working.",
        "Positive signals detected. This is a good time to consolidate your momentum.",
        "Great energy detected! Channel this momentum into what matters to you.",
    ],
    "negative": [
        "Difficult moments happen. Acknowledging them is a constructive first step.",
        "Your message suggests some tension. Take it one step at a time.",
        "Expressing what you feel is already progress. Be patient with yourself.",
        "Be gentle with yourself today. Give yourself room to rest and recharge.",
    ],
    "neutral": [
        "A balanced day is valuable in itself.",
        "Your state appears stable. A good opportunity to recharge.",
        "A calm period can be a good time to reflect and plan.",
        "Neutral days provide a steady baseline to organize your thoughts.",
    ],
}

action_plans = {
    "positive": {
        "plan": [
            "Share your positive energy with someone close to you.",
            "Note three inspiring wins or moments from today.",
            "Dedicate time to a creative activity or project you genuinely enjoy.",
            "Consolidate your progress and plan a rewarding milestone.",
        ],
        "activities": [
            ("menu_book", "Read a few pages of an inspiring book or article"),
            ("headphones", "Listen to an uplifting mood playlist"),
            ("directions_walk", "Take a mindful 15-minute outdoor walk"),
            ("mail", "Send a warm message of appreciation to a friend"),
            ("celebration", "Take 5 minutes to celebrate a recent personal success"),
            ("self_improvement", "Write down your top goals for the upcoming week"),
            ("palette", "Engage in a short creative drawing or writing session"),
        ],
    },
    "negative": {
        "plan": [
            "Pause, drop your shoulders, and breathe slowly for a few minutes.",
            "Write your thoughts down on paper to gain objective distance.",
            "Remember that emotional states are temporary and constantly evolving.",
            "Lower immediate expectations and focus solely on essential needs.",
        ],
        "activities": [
            ("headphones", "Listen to calm ambient sounds or nature acoustics"),
            ("edit_note", "Journal freely for 5 minutes without filter"),
            ("park", "Step outside to get fresh air and sunshine"),
            ("call", "Reach out to someone you trust"),
            ("spa", "Practice box breathing (4s inhale, 4s hold, 4s exhale)"),
            ("local_cafe", "Enjoy a warm beverage in total calm"),
            ("bed", "Allow yourself a silent rest or 15-minute nap"),
        ],
    },
    "neutral": {
        "plan": [
            "Use this calm, balanced period to restore baseline energy.",
            "Try a creative or manual activity without any pressure.",
            "Plan something pleasant and organized for tomorrow.",
            "Use this steady baseline to organize your space and thoughts.",
        ],
        "activities": [
            ("extension", "Do a puzzle or short brain-teaser"),
            ("play_circle", "Watch an uplifting or educational video"),
            ("coffee", "Take a quiet, screen-free break"),
            ("self_improvement", "Try a short 5-minute grounded meditation"),
            ("cleaning_services", "Tidy or declutter your immediate desk or workspace"),
            ("menu_book", "Read a chapter of an informative book"),
            ("checklist", "Set one realistic intention for the rest of the day"),
        ],
    },
}

emotion_keywords = {
    "fatigue": ["fatigue", "tired", "exhausted", "drained", "sleepy", "epuise", "creve"],
    "joy": ["happy", "joyful", "glad", "excited", "heureux", "content", "joie"],
    "anger": ["angry", "mad", "furious", "annoyed", "colere", "enerve", "fache"],
    "stress": ["stress", "stressed", "anxious", "tense", "stresse", "anxieux", "pression"],
    "sadness": ["sad", "down", "blue", "melancholic", "triste", "deprime", "solitude"],
}

# --------------------------------------------------
# Header with Purple Accent Style & Restored Cloud Icon
# --------------------------------------------------
hour = datetime.now().hour
greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"

st.markdown(
    f"""
    <div class="mt-header">
        <div class="mt-greeting">{greeting}</div>
        <div class="mt-title">💬 MindTalk</div>
        <div class="mt-sub">A short daily check-in on your mood.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

main_col, side_col = st.columns([2, 1], gap="large")

# ==================================================
# LEFT COLUMN
# ==================================================
with main_col:
    st.subheader("Mood check-in", anchor=False)
    
    # Text input area with clear option
    user_input = st.text_area(
        "How are you feeling today?",
        value=st.session_state.user_input_val,
        placeholder="Describe your day, your thoughts, or how you feel...",
        height=130,
        max_chars=1000,
        key="current_user_input"
    )

    auto_clear = st.checkbox("Auto-clear description box after analysis", value=False)

    col1, col2, col3 = st.columns([2, 1, 1])
    analyze = col1.button("Analyse my mood", icon=":material/psychology:", type="primary", use_container_width=True)
    clear = col2.button("Clear", icon=":material/delete:", use_container_width=True)

    if st.session_state.history:
        col3.download_button(
            "Export",
            pd.DataFrame(st.session_state.history).to_csv(index=False),
            "mindtalk_history.csv",
            "text/csv",
            icon=":material/download:",
            use_container_width=True,
        )
    else:
        col3.button("Export", icon=":material/download:", disabled=True, use_container_width=True)

    if clear:
        st.session_state.history.clear()
        st.session_state.plan = None
        st.session_state.activities = []
        st.session_state.user_input_val = ""
        st.toast("History cleared.", icon=":material/check_circle:")
        st.rerun()

    # ---------------- Analysis ----------------
    if analyze:
        if not user_input.strip():
            st.warning("Please write a few words first.", icon=":material/edit:")
        else:
            try:
                compound = analyze_hybrid_sentiment(user_input)
                timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")

                if compound >= 0.05:
                    mood_key = "positive"
                elif compound <= -0.05:
                    mood_key = "negative"
                else:
                    mood_key = "neutral"

                mood = MOODS[mood_key]
                feedback = random.choice(feedback_messages[mood_key])

                with st.container(border=True):
                    st.markdown(f"**{mood['icon']} Overall sentiment: {mood['label']}**")
                    st.write(feedback)
                    st.progress((compound + 1) / 2, text=f"Sentiment score: {compound:+.2f}")

                    text = normalize_text(user_input.lower())
                    for emotion, words in emotion_keywords.items():
                        if any(w in text for w in words):
                            st.caption(f"Dominant emotion detected: **{emotion.capitalize()}**")
                            break

                plan_data = action_plans[mood_key]
                st.session_state.plan = random.choice(plan_data["plan"])
                
                sample_count = min(3, len(plan_data["activities"]))
                st.session_state.activities = random.sample(plan_data["activities"], sample_count)

                st.session_state.history.append(
                    {
                        "text": user_input,
                        "category": mood["label"],
                        "mood_key": mood_key,
                        "feedback": feedback,
                        "compound": compound,
                        "time": timestamp,
                    }
                )

                if auto_clear:
                    st.session_state.user_input_val = ""
                    st.rerun()

            except Exception:
                st.error("Something went wrong during the analysis.", icon=":material/error:")

    # ---------------- Statistics & History (With Graph) ----------------
    if st.session_state.history:
        history = st.session_state.history

        st.divider()
        st.subheader("Overview", anchor=False)

        counts = {k: sum(h["mood_key"] == k for h in history) for k in MOODS}
        m1, m2, m3 = st.columns(3)
        m1.metric("Positive", counts["positive"])
        m2.metric("Neutral", counts["neutral"])
        m3.metric("Negative", counts["negative"])

        # Interactive Graph Representation
        st.write("")
        df_chart = pd.DataFrame(history)
        df_chart["Entry"] = range(1, len(df_chart) + 1)

        fig = px.line(
            df_chart,
            x="Entry",
            y="compound",
            markers=True,
            title="Sentiment Progression Curve",
            labels={"compound": "Sentiment Score", "Entry": "Check-in Index"},
            color_discrete_sequence=[ACCENT],
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            margin=dict(t=30, b=10, l=10, r=10),
            height=220,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        
        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Recent entries", anchor=False)

        for entry in reversed(history[-5:]):
            mood = MOODS[entry["mood_key"]]
            safe_text = entry["text"].replace("<", "&lt;").replace(">", "&gt;")
            st.markdown(
                f"""
                <div class="mt-entry" style="border-left-color:{mood['color']};">
                    <span class="mt-badge" style="background:{mood['color']};">{entry['category']}</span>
                    <span class="meta">&nbsp;{entry['time']}</span>
                    <div style="margin-top:0.4rem;">{safe_text}</div>
                    <div class="meta" style="margin-top:0.3rem;">{entry['feedback']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ==================================================
# RIGHT COLUMN
# ==================================================
with side_col:
    st.subheader("Recommendations", anchor=False)

    if st.session_state.plan:
        st.markdown("**Suggested focus**")
        st.markdown(f"<div class='mt-plan'>{st.session_state.plan}</div>", unsafe_allow_html=True)

        st.markdown("&nbsp;")
        st.markdown("**Suggested activities**")
        for icon, label in st.session_state.activities:
            with st.container(border=True):
                st.markdown(f":material/{icon}: &nbsp; {label}")
    else:
        st.info("Run an analysis to receive tailored suggestions.", icon=":material/info:")

    st.markdown(
        "<div class='mt-note'>MindTalk is a self-reflection tool. It does not provide medical "
        "advice or diagnosis. If you are struggling, please consider speaking with a professional.</div>",
        unsafe_allow_html=True,
    )