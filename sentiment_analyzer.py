import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime

# Télécharger le lexique de sentiment si nécessaire
nltk.download('vader_lexicon')

# Initialiser l'analyseur de sentiments
sia = SentimentIntensityAnalyzer()

# Configuration de la page
st.set_page_config(page_title="Feelings Companion", page_icon="💬", layout="centered")

# --- Titre ---
st.title("💬 Feelings Companion")
st.write("Prends un moment pour toi et partage ce que tu ressens. Je vais t’apporter un petit mot réconfortant 🌈")

# --- Zone de saisie ---
user_input = st.text_area("Comment te sens-tu aujourd’hui ?", placeholder="Écris ton ressenti ici...")

# Historique
if "history" not in st.session_state:
    st.session_state.history = []

# --- Bouton d’analyse ---
if st.button("Analyser mon humeur"):
    if user_input.strip():
        sentiment = sia.polarity_scores(user_input)
        compound = sentiment['compound']
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M")

        # Détermination du sentiment avec textes humains
        if compound >= 0.05:
            category = "😊 Positif"
            feedback = "Super ! C’est génial de te sentir bien. Profite de cette belle énergie ✨"
            st.success(feedback)
        elif compound <= -0.05:
            category = "😞 Négatif"
            feedback = "Je comprends, les jours difficiles font partie de la vie 💛 Prends soin de toi et respire doucement."
            st.error(feedback)
        else:
            category = "😐 Neutre"
            feedback = "Une journée calme et équilibrée 🍃. Parfois, se sentir tranquille est déjà un vrai cadeau."
            st.info(feedback)

        # Feedback personnalisé rapide
        text = user_input.lower()
        if "tired" in text or "fatigué" in text:
            st.write("💤 Tu sembles fatigué. Prends un petit moment pour toi.")
        elif "happy" in text or "heureux" in text:
            st.write("🌞 Ça fait plaisir de te voir heureux(se) ! Continue à sourire 💕")
        elif "angry" in text or "énervé" in text:
            st.write("😤 Respire un bon coup et prends du recul. Ça ira mieux ensuite.")

        # Barre de progression
        st.progress((compound + 1) / 2)

        # Enregistrer dans l’historique avec date/heure
        st.session_state.history.append({
            "text": user_input,
            "category": category,
            "feedback": feedback,
            "time": current_time
        })
    else:
        st.warning("Écris quelque chose avant d’analyser ton sentiment 😉")

# --- Historique des analyses ---
if st.session_state.history:
    st.subheader("🕓 Nos dernieres interactions")
    for entry in reversed(st.session_state.history[-5:]):
        # Design strictement conservé, juste ajout de la date
        st.markdown(f"**🗣️ {entry['text']}**  →  {entry['category']}  <br>💡 _{entry['feedback']}_  <br>🕒 {entry['time']}", unsafe_allow_html=True)
