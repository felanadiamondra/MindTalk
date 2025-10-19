import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime
import pandas as pd
import random

# Télécharger le lexique de sentiment si nécessaire
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
st.set_page_config(page_title="MindTalk", page_icon="💬", layout="centered")

# --- Variables initiales ---
category = ""
mood_key = ""
plan = ""
activities = []

# --- Variations de feedback ---
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

# --- Plans d'action et activités selon l'humeur ---
action_plans = {
    "positive": {
        "plan": [
            "Continue à nourrir ton énergie positive en partageant un moment avec quelqu’un que tu aimes 💕",
            "Note trois choses qui t’ont rendu(e) heureux(se) aujourd’hui ✨",
            "Fais quelque chose que tu adores pour prolonger cette belle humeur : écoute de la musique, danse ou cuisine 🥗"
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
            "Essaie de mettre des mots sur ce que tu ressens, ça aide à libérer le mental 💭",
            "Souviens-toi : toutes les émotions sont temporaires, tu n’es pas seul(e) 💛"
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

# --- Message de bienvenue ---
hour = datetime.now().hour
if hour < 12:
    st.markdown("☀️ **Bonjour ! Prêt(e) pour une nouvelle journée ?**")
elif hour < 18:
    st.markdown("🌤 **Bon après-midi ! Comment ça va ?**")
else:
    st.markdown("🌙 **Bonsoir ! Une petite réflexion sur ta journée ?**")

st.markdown("<h1 style='text-align:center; color:#9B6BA7;'>💬 MindTalk</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#4D87A6;'>Prends un moment pour toi et partage ce que tu ressens. Je vais t’apporter un petit mot réconfortant 🌈</p>", unsafe_allow_html=True)

# --- Zone de texte ---
user_input = st.text_area("Comment te sens-tu aujourd’hui ?", placeholder="Écris ton ressenti ici...")

# --- Historique ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Boutons côte à côte ---
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    analyze_button = st.button("Analyser mon humeur")
with col2:
    clear_button = st.button("🗑 Vider l'historique")
with col3:
    if st.session_state.history:
        st.download_button(
            label="📥 Télécharger",
            data=pd.DataFrame(st.session_state.history).to_csv(index=False),
            file_name="mindtalk_history.csv",
            mime="text/csv"
        )
    else:
        st.button("📥 Télécharger", disabled=True)

# --- Vider l'historique ---
if clear_button:
    st.session_state.history = []
    st.success("Historique vidé avec succès ! ✨")

# --- Analyse de l’humeur ---
if analyze_button:
    if user_input.strip():
        sentiment = sia.polarity_scores(user_input)
        compound = sentiment['compound']
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M")

        # Catégorie
        if compound >= 0.05:
            category = "😊 Positif"
            feedback = random.choice(positive_responses)
            st.success(f"💡 {feedback}")
        elif compound <= -0.05:
            category = "😞 Négatif"
            feedback = random.choice(negative_responses)
            st.error(f"💡 {feedback}")
        else:
            category = "😐 Neutre"
            feedback = random.choice(neutral_responses)
            st.info(f"💡 {feedback}")

        # Feedback rapide contextuel
        text = user_input.lower()
        if "tired" in text or "fatigué" in text:
            st.write("💤 Tu sembles fatigué. Prends un petit moment pour toi.")
        elif "happy" in text or "heureux" in text:
            st.write("🌞 Ça fait plaisir de te voir heureux(se) ! Continue à sourire 💕")
        elif "angry" in text or "énervé" in text:
            st.write("😤 Respire un bon coup et prends du recul. Ça ira mieux ensuite.")
        elif "stress" in text or "stressé" in text:
            st.write("🧘 Prends 2 minutes pour respirer profondément. Inspire… expire…")
        elif "triste" in text or "sad" in text:
            st.write("🌱 Écris une chose qui t’a fait sourire aujourd’hui, même petite.")

        # Barre de progression
        st.progress((compound + 1) / 2)
        st.caption(f"**Sentiment global : {category}**")

        # Plan et activités
        if "Positif" in category:
            mood_key = "positive"
        elif "Négatif" in category:
            mood_key = "negative"
        else:
            mood_key = "neutral"

        plan = random.choice(action_plans[mood_key]["plan"])
        activities = random.sample(action_plans[mood_key]["activities"], 2)

        # Historique
        st.session_state.history.append({
            "text": user_input,
            "category": category,
            "feedback": feedback,
            "time": current_time,
            "compound": compound
        })
    else:
        st.warning("Écris quelque chose avant d’analyser ton sentiment 😉")

# --- Résumé statistique ---
if st.session_state.history:
    st.subheader("📊 Résumé de vos humeurs")
    pos = sum(1 for h in st.session_state.history if "😊 Positif" in h["category"])
    neg = sum(1 for h in st.session_state.history if "😞 Négatif" in h["category"])
    neu = sum(1 for h in st.session_state.history if "😐 Neutre" in h["category"])
    st.write(f"😊 Positif : {pos} | 😞 Négatif : {neg} | 😐 Neutre : {neu}")

# --- Historique des interactions ---
if st.session_state.history:
    st.subheader("🕓 Dernières interactions")
    for entry in reversed(st.session_state.history[-10:]):
        st.markdown(
            f"**🗣️ {entry['text']}**  →  {entry['category']}  <br>💡 _{entry['feedback']}_  <br>🕒 {entry['time']}",
            unsafe_allow_html=True
        )

# --- Plan d’action et activités ---
if plan:
    st.markdown("### 🧭 Plan d’action du jour")
    st.write(plan)

if activities:
    st.markdown("### 🎯 Activités que tu peux faire")
    for activity in activities:
        st.write(f"- {activity}")
