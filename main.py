import random
import time
import unicodedata
from datetime import datetime

import nltk
import pandas as pd
import plotly.express as px
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer

# --------------------------------------------------
# 1. Configuration & Custom CSS
# --------------------------------------------------
st.set_page_config(
    page_title="MindTalk — Mood & Mind Tracker",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .hero-container {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        padding: 2rem;
        border-radius: 16px;
        color: #FFFFFF !important;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.3);
    }
    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
        color: #FFFFFF !important;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.9) !important;
    }
    .custom-card {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .history-card {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #6366F1;
        border-top: 1px solid rgba(128, 128, 128, 0.15);
        border-right: 1px solid rgba(128, 128, 128, 0.15);
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 0.75rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 2. Advanced Multi-lingual Sentiment & Emotion Engine
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_analyzer():
    try:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception:
        st.error("⚠️ Impossible d'initialiser le moteur NLTK.")
        st.stop()

sia = load_sentiment_analyzer()

# French Sentiment & Emotion Dictionary
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
        "furieuse", "rage", "haine", "agacé", "agacée", "irrité", "irritée", "injuste"
    ],
    "stress": [
        "stress", "stresse", "stressé", "stressée", "anxieux", "anxieuse", "pression", "panique",
        "overdose", "submergé", "submergée", "dépassé", "dépassée", "tendu", "tendue"
    ],
    "fatigue": [
        "fatigue", "fatigué", "fatiguée", "épuisé", "épuisée", "crevé", "crevée", "draine",
        "drainé", "drainée", "hs", "sommeil", "lassitude", "à bout"
    ]
}

INTENSIFIERS = ["très", "tres", "trop", "tellement", "vraiment", "extrêmement", "extremement", "super"]

def analyze_hybrid_sentiment(text: str):
    """
    Combines VADER (English) with a rich custom French NLP Lexicon & Intensity Calculator
    """
    normalized = unicodedata.normalize("NFD", text.lower()).encode("ascii", "ignore").decode("utf-8")
    words = normalized.split()
    
    # 1. English VADER Score
    vader_scores = sia.polarity_scores(text)
    vader_compound = vader_scores["compound"]
    
    # 2. French Lexicon Score & Emotion Detector
    pos_count = 0
    neg_count = 0
    emotion_scores = {"Tristesse": 0, "Joie": 0, "Colère": 0, "Stress": 0, "Fatigue": 0}
    
    multiplier = 1.0
    for i, word in enumerate(words):
        # Check intensifiers (ex: "très triste")
        if word in INTENSIFIERS:
            multiplier = 1.5
            continue
            
        if word in FRENCH_LEXICON["positive"]:
            pos_count += 1 * multiplier
            emotion_scores["Joie"] += 1 * multiplier
        elif word in FRENCH_LEXICON["negative"]:
            neg_count += 1 * multiplier
            emotion_scores["Tristesse"] += 1 * multiplier
            
        if word in FRENCH_LEXICON["anger"]:
            emotion_scores["Colère"] += 1 * multiplier
        if word in FRENCH_LEXICON["stress"]:
            emotion_scores["Stress"] += 1 * multiplier
        if word in FRENCH_LEXICON["fatigue"]:
            emotion_scores["Fatigue"] += 1 * multiplier
            
        multiplier = 1.0 # reset multiplier after applied

    # Check capitalization (ALL CAPS = Higher Intensity)
    if text.isupper() and len(text) > 5:
        pos_count *= 1.3
        neg_count *= 1.3

    # Check Exclamation / Question marks
    if "!" in text or "?" in text:
        pos_count *= 1.1
        neg_count *= 1.1

    # Hybrid Compound Score Calculation (-1.0 to +1.0)
    fr_total = pos_count + neg_count
    if fr_total > 0:
        fr_compound = (pos_count - neg_count) / fr_total
    else:
        fr_compound = 0.0

    # Fusion des scores VADER + Lexique Français
    if fr_total > 0:
        final_compound = fr_compound
    else:
        final_compound = vader_compound

    # Category Determination
    if final_compound >= 0.15:
        category, emoji, mood_key = "Positif", "😊", "positive"
    elif final_compound <= -0.15:
        category, emoji, mood_key = "Négatif", "😞", "negative"
    else:
        category, emoji, mood_key = "Neutre", "😐", "neutral"

    # Top Emotion Selection
    top_emotion = max(emotion_scores, key=emotion_scores.get)
    if emotion_scores[top_emotion] == 0:
        top_emotion = "Équilibré"

    return {
        "compound": round(final_compound, 2),
        "category": category,
        "emoji": emoji,
        "mood_key": mood_key,
        "top_emotion": top_emotion,
        "emotion_scores": emotion_scores
    }

# --------------------------------------------------
# 3. Session State
# --------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None
if "user_text_input" not in st.session_state:
    st.session_state.user_text_input = ""
if "completed_activities" not in st.session_state:
    st.session_state.completed_activities = set()

# --------------------------------------------------
# 4. Action Plans & Prompts
# --------------------------------------------------
action_plans = {
    "positive": {
        "plan": "Partagez cette énergie positive avec votre entourage 💕",
        "activities": [
            "📖 Lire un chapitre d'un livre inspirant",
            "🎶 Écouter votre playlist préférée",
            "🚶 Faire une marche consciente de 15 min",
            "💌 Envoyer un message de gratitude à un ami",
        ],
    },
    "negative": {
        "plan": "Prenez une pause, relâchez les épaules et respirez profondément 🧘",
        "activities": [
            "🎧 Écouter des sons relaxants d'ambiance",
            "✍️ Écrire librement dans un journal (5 min)",
            "🌿 Sortir prendre l'air frais quelques instants",
            "📞 Appeler ou envoyer un message à un proche de confiance",
        ],
    },
    "neutral": {
        "plan": "Profitez de ce calme pour recharger vos batteries 🍃",
        "activities": [
            "🧩 Faire une petite énigme ou un casse-tête",
            "📺 Regarder une vidéo courte et inspirante",
            "☕ Déguster une boisson chaude en pleine conscience",
            "🧘 Faire 3 minutes de méditation guidée",
        ],
    },
}

quick_prompts = [
    "Je me sens vraiment triste parce que...",
    "Aujourd'hui je suis très motivé et...",
    "Je ressens beaucoup de stress avec...",
    "Je suis épuisé et j'ai besoin de...",
]

# Header
hour = datetime.now().hour
greeting_prefix = "Bonjour" if hour < 12 else "Bon après-midi" if hour < 18 else "Bonsoir"

st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-title">💬 MindTalk</div>
        <div class="hero-subtitle">{greeting_prefix} ! Prenez un instant pour faire le point sur votre état d'esprit.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 5. Core Layout
# --------------------------------------------------
main_col, side_col = st.columns([2.2, 1], gap="large")

with main_col:
    st.subheader("💬 Exprimer votre ressenti")

    st.caption("⚡ **Amorces rapides d'expression :**")
    chip_cols = st.columns(len(quick_prompts))
    for idx, prompt in enumerate(quick_prompts):
        if chip_cols[idx].button(prompt, key=f"prompt_{idx}", use_container_width=True):
            st.session_state.user_text_input = prompt + " "

    with st.form(key="mood_form", clear_on_submit=False):
        user_input = st.text_area(
            label="Qu'avez-vous sur le cœur ou en tête ?",
            value=st.session_state.user_text_input,
            placeholder="Exemple : Je me sens tellement triste et fatigué aujourd'hui...",
            height=120,
            max_chars=1000,
        )
        
        c_left, c_right = st.columns([2, 1])
        with c_left:
            submit_button = st.form_submit_button("✨ Analyser mon humeur", type="primary", use_container_width=True)
        with c_right:
            clear_input = st.form_submit_button("🧹 Effacer le texte", use_container_width=True)

    if clear_input:
        st.session_state.user_text_input = ""
        st.rerun()

    if submit_button:
        if not user_input.strip():
            st.warning("⚠️ Veuillez écrire quelques mots avant de lancer l'analyse.")
        else:
            with st.status("Analyse sémantique hybride en cours...", expanded=False) as status:
                time.sleep(0.2)
                
                # Hybrid NLP Model Call
                res_nlp = analyze_hybrid_sentiment(user_input)
                timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")

                plan = action_plans[res_nlp["mood_key"]]["plan"]
                activities = action_plans[res_nlp["mood_key"]]["activities"]

                analysis_result = {
                    "id": len(st.session_state.history) + 1,
                    "text": user_input,
                    "category": res_nlp["category"],
                    "emoji": res_nlp["emoji"],
                    "compound": res_nlp["compound"],
                    "emotion": res_nlp["top_emotion"],
                    "emotion_scores": res_nlp["emotion_scores"],
                    "plan": plan,
                    "activities": activities,
                    "time": timestamp,
                }

                st.session_state.last_analysis = analysis_result
                st.session_state.history.append(analysis_result)
                status.update(label="Analyse terminée !", state="complete")
                st.toast("Ressenti analysé et enregistré !", icon="🧠")

    # Immediate Breakdown Output
    if st.session_state.last_analysis:
        res = st.session_state.last_analysis
        st.markdown("---")
        
        st.subheader("🎯 Résultat de l'analyse enrichie")
        m1, m2, m3 = st.columns(3)
        m1.metric("Tonalité générale", f"{res['emoji']} {res['category']}")
        m2.metric("Score de sentiment", f"{res['compound']:.2f}")
        m3.metric("Émotion dominante", res["emotion"])

        gauge_val = float((res["compound"] + 1) / 2)
        st.progress(gauge_val)
        st.caption("Polarité émotionnelle (-1.0 Très Négatif ↔ +1.0 Très Positif)")

    # Tabs for History & Analytics
    st.markdown("---")
    tab_analytics, tab_history = st.tabs(["📊 Graphiques & Tendances", "📜 Journal & Filtres"])

    with tab_analytics:
        if st.session_state.history:
            df_hist = pd.DataFrame(st.session_state.history)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Entrées", len(df_hist))
            c2.metric("Positives", len(df_hist[df_hist["category"] == "Positif"]))
            c3.metric("Neutres", len(df_hist[df_hist["category"] == "Neutre"]))
            c4.metric("Négatives", len(df_hist[df_hist["category"] == "Négatif"]))

            st.write("")

            fig = px.pie(
                df_hist,
                names="category",
                title="Répartition des humeurs enregistrées",
                color="category",
                color_discrete_map={
                    "Positif": "#10B981",
                    "Neutre": "#9CA3AF",
                    "Négatif": "#EF4444",
                },
                hole=0.4,
            )
            fig.update_layout(
                margin=dict(t=40, b=0, l=0, r=0),
                height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="gray"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune donnée enregistrée pour le moment.")

    with tab_history:
        if st.session_state.history:
            f_col1, f_col2 = st.columns([2, 1])
            with f_col1:
                search_query = st.text_input("🔍 Rechercher un mot-clé", placeholder="Ex: triste, fatigue...")
            with f_col2:
                category_filter = st.selectbox("Filtrer par catégorie", ["Toutes", "Positif", "Neutre", "Négatif"])

            filtered_history = st.session_state.history
            if category_filter != "Toutes":
                filtered_history = [h for h in filtered_history if h["category"] == category_filter]
            if search_query.strip():
                filtered_history = [h for h in filtered_history if search_query.lower() in h["text"].lower()]

            t_col1, t_col2 = st.columns([1, 1])
            with t_col1:
                df_export = pd.DataFrame(st.session_state.history)[["time", "category", "compound", "emotion", "text"]]
                st.download_button(
                    label="📥 Exporter en CSV",
                    data=df_export.to_csv(index=False),
                    file_name="mindtalk_journal.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with t_col2:
                if st.button("🗑️ Effacer le journal", use_container_width=True):
                    st.session_state.history.clear()
                    st.session_state.last_analysis = None
                    st.toast("Historique réinitialisé !", icon="🧹")
                    st.rerun()

            st.write("")
            if filtered_history:
                for entry in reversed(filtered_history[-5:]):
                    st.markdown(
                        f"""
                        <div class="history-card">
                            <small style="opacity: 0.7;">{entry['time']} | Émotion : <b>{entry['emotion']}</b></small><br>
                            <span style="font-size: 1.05rem;">{entry['emoji']} <b>{entry['category']}</b> (Score : {entry['compound']:.2f})</span>
                            <p style="margin-top: 5px; margin-bottom: 0px;">"{entry['text']}"</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("Aucun résultat ne correspond à vos filtres.")
        else:
            st.info("Le journal d'historique est actuellement vide.")

# --------------------------------------------------
# 6. Sidebar Guidance & Checklist
# --------------------------------------------------
with side_col:
    st.subheader("🧭 Plan d'action interactif")

    if st.session_state.last_analysis:
        res = st.session_state.last_analysis

        st.markdown(
            f"""
            <div class="custom-card">
                <h4 style="margin-top: 0; color: #818CF8;">💡 État d'esprit recommandé</h4>
                <p style="font-size: 0.95rem; margin-bottom: 0;">{res['plan']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🎯 To-Do bien-être")
        st.caption("Cochez les activités au fur et à mesure :")

        for idx, act in enumerate(res["activities"]):
            act_key = f"act_{res['id']}_{idx}"
            is_checked = act_key in st.session_state.completed_activities
            
            if st.checkbox(act, value=is_checked, key=act_key):
                if act_key not in st.session_state.completed_activities:
                    st.session_state.completed_activities.add(act_key)
                    st.toast(f"Bravo ! Activité faite : {act}", icon="🎉")
            else:
                st.session_state.completed_activities.discard(act_key)

        completed_count = sum(1 for idx in range(len(res["activities"])) if f"act_{res['id']}_{idx}" in st.session_state.completed_activities)
        total_count = len(res["activities"])
        st.write("")
        st.progress(completed_count / total_count if total_count > 0 else 0)
        st.caption(f"Progression : {completed_count}/{total_count} activité(s)")

    else:
        st.markdown(
            """
            <div class="custom-card" style="text-align: center; opacity: 0.8;">
                <p style="margin-bottom: 0;">Analysez votre humeur pour débloquer votre plan d'action interactif.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )