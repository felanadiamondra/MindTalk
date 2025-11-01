import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime
import pandas as pd
import random
import unicodedata

# --- Configuration ---
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
st.set_page_config(page_title="MindTalk", page_icon="💬", layout="wide")

# --- Thème pastel ---
PRIMARY_COLOR = "#9B6BA7"
SECONDARY_COLOR = "#4D87A6"
BG_LIGHT = "#FAF7F2"
BG_GREY = "#63605f"

# --- Variables ---
category = ""
mood_key = ""
plan = ""
activities = []

# --- Listes de feedback ---
positive_responses = [
    "🌞 Super ! C’est génial de te sentir bien. Profite de cette belle énergie ✨",
    "💪 Garde cette belle humeur, elle est contagieuse 😄",
    "🌸 Quelle joie de lire ça ! Continue à répandre ces bonnes ondes 🌈",
    "😄 Excellent ! Garde le sourire, il illumine ta journée ☀️"
]

negative_responses = [
    "💛 Je comprends, les jours difficiles font partie de la vie. Prends soin de toi 💫",
    "🌧 Même les nuages passent… demain sera plus doux 🌤",
    "🤗 Courage, un petit pas à la fois. Tu vas t’en sortir 💕",
    "💬 Tu as bien fait d’en parler. Lâcher un peu, c’est déjà avancer 🌱"
]

neutral_responses = [
    "🍃 Une journée calme et équilibrée, c’est déjà précieux.",
    "🕊 Parfois, ne rien ressentir de particulier, c’est un signe d’apaisement.",
    "🌼 Une journée neutre, c’est peut-être le calme avant de belles choses 🌈",
    "☕ Un moment tranquille, parfait pour se recentrer un peu."
]

# --- Plans et activités ---
action_plans = {
    "positive": {
        "plan": [
            "Continue à nourrir ton énergie positive en partageant un moment avec quelqu’un que tu aimes 💕",
            "Note trois choses qui t’ont rendu(e) heureux(se) aujourd’hui ✨",
            "Fais quelque chose que tu adores : musique, danse, cuisine 🥗"
        ],
        "activities": [
            "📖 Lire un livre inspirant",
            "🎶 Écouter ta playlist préférée",
            "🚶 Sortir prendre l’air ou marcher 10 minutes",
            "💌 Envoyer un message gentil à quelqu’un"
        ]
    },
    "negative": {
        "plan": [
            "Prends un moment pour respirer profondément et te recentrer 🧘",
            "Essaie de mettre des mots sur ce que tu ressens 💭",
            "Souviens-toi : toutes les émotions sont temporaires 💛"
        ],
        "activities": [
            "🎧 Écouter une musique douce",
            "✍️ Écrire dans un journal ce que tu ressens",
            "🌿 Faire une promenade au calme",
            "📞 Appeler un ami de confiance"
        ]
    },
    "neutral": {
        "plan": [
            "Profite de cette stabilité pour te ressourcer 🍃",
            "Essaye une activité nouvelle pour stimuler ta créativité 🎨",
            "Planifie quelque chose qui te fait plaisir pour demain 🌅"
        ],
        "activities": [
            "🧩 Faire un puzzle ou un jeu de logique",
            "📺 Regarder un film inspirant",
            "☕ Prendre un moment de calme sans téléphone",
            "🧘 Essayer une courte méditation guidée"
        ]
    }
}

emotion_keywords = {
    "fatigue": ["fatigue", "fatiguer", "tired", "epuise", "creve", "lasse", "extenuer"],
    "joie": ["happy", "heureux", "contente", "joyeux", "satisfait", "ravi"],
    "colere": ["angry", "enerve", "furieux", "vexe", "agace"],
    "stress": ["stress", "stresse", "angoisse", "pression", "tendu"],
    "tristesse": ["triste", "sad", "deprime", "chagrin", "morose", "melancolique"]
}

def normalize_text(text):
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    return text

# --- En-tête ---
hour = datetime.now().hour
if hour < 12:
    greeting = "☀️ Bonjour ! Prêt(e) pour une nouvelle journée ?"
elif hour < 18:
    greeting = "🌤 Bon après-midi ! Comment ça va ?"
else:
    greeting = "🌙 Bonsoir ! Une petite réflexion sur ta journée ?"

st.markdown(f"<h2 style='color:{PRIMARY_COLOR}; text-align:center'>{greeting}</h2>", unsafe_allow_html=True)
st.markdown(f"<h1 style='text-align:center; color:{SECONDARY_COLOR}'>💬 MindTalk</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555'>Prends un moment pour toi et partage ce que tu ressens 🌈</p>", unsafe_allow_html=True)

# --- Séparation layout (2 colonnes principales) ---
main_col, side_col = st.columns([2, 1])

# --------------------------- 🧠 COLONNE GAUCHE ---------------------------
with main_col:
    st.markdown("---")
    st.subheader("📝 Ton ressenti du jour")
    user_input = st.text_area("Exprime-toi librement :", placeholder="Écris ton ressenti ici...", height=120)

    if "history" not in st.session_state:
        st.session_state.history = []

    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    with col_btn1:
        analyze_button = st.button("💬 Analyser mon humeur")
    with col_btn2:
        clear_button = st.button("🗑 Vider l'historique")
    with col_btn3:
        if st.session_state.history:
            st.download_button(
                label="📥 Télécharger",
                data=pd.DataFrame(st.session_state.history).to_csv(index=False),
                file_name="mindtalk_history.csv",
                mime="text/csv"
            )
        else:
            st.button("📥 Télécharger", disabled=True)

    if clear_button:
        st.session_state.history = []
        st.success("Historique vidé avec succès ! ✨")

    # --- Analyse principale ---
    if analyze_button:
        if user_input.strip():
            sentiment = sia.polarity_scores(user_input)
            compound = sentiment["compound"]
            current_time = datetime.now().strftime("%d/%m/%Y %H:%M")

            if compound >= 0.05:
                category = "😊 Positif"
                feedback = random.choice(positive_responses)
                st.success(feedback)
                mood_key = "positive"
            elif compound <= -0.05:
                category = "😞 Négatif"
                feedback = random.choice(negative_responses)
                st.error(feedback)
                mood_key = "negative"
            else:
                category = "😐 Neutre"
                feedback = random.choice(neutral_responses)
                st.info(feedback)
                mood_key = "neutral"

            # Barre et ressenti contextuel
            st.progress((compound + 1) / 2)
            st.caption(f"**Sentiment global : {category}**")

            text = normalize_text(user_input.lower())
            for key, words in emotion_keywords.items():
                if any(w in text for w in words):
                    if key == "fatigue":
                        st.write("💤 Tu sembles fatigué. Prends un moment pour toi.")
                    elif key == "joie":
                        st.write("🌞 Ça fait plaisir de te voir heureux(se) ! Continue à sourire 💕")
                    elif key == "colere":
                        st.write("😤 Respire un bon coup et prends du recul. Ça ira mieux ensuite.")
                    elif key == "stress":
                        st.write("🧘 Inspire profondément... expire... tout va bien 🌿")
                    elif key == "tristesse":
                        st.write("🌱 Écris une chose qui t’a fait sourire aujourd’hui.")

            # Génération plan & activités
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
            st.warning("✏️ Écris quelque chose avant d’analyser ton sentiment !")

    # --- Résumé & Historique ---
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📊 Statistiques d’humeur")
        pos = sum(1 for h in st.session_state.history if "😊 Positif" in h["category"])
        neg = sum(1 for h in st.session_state.history if "😞 Négatif" in h["category"])
        neu = sum(1 for h in st.session_state.history if "😐 Neutre" in h["category"])
        st.write(f"**😊 Positif : {pos} | 😞 Négatif : {neg} | 😐 Neutre : {neu}**")

        st.markdown("---")
        st.subheader("🕓 Historique récent")
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

# --------------------------- 💡 COLONNE DROITE ---------------------------
with side_col:
    st.markdown("---")
    st.subheader("🧭 Plan d’action & Activités")
    if plan or activities:
        if plan:
            st.markdown(f"<div style='background-color:{BG_GREY}; padding:10px; border-radius:10px;'>💡 {plan}</div>", unsafe_allow_html=True)
        if activities:
            st.markdown("### 🎯 Activités suggérées")
            for activity in activities:
                st.markdown(f"- {activity}")
    else:
        st.info("Analyse ton humeur pour découvrir des activités adaptées 💬")
