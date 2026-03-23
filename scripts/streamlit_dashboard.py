import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

from utils.scoring_utils import *
from utils.explainability_utils import explain_prediction


# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="News Popularity AI",
    page_icon="📰",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
body { background-color: #0e1117; }
h1, h2, h3 { color: white; }
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    margin: 10px 0px;
    box-shadow: 0px 0px 15px rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD ----------------
model = joblib.load("models/popularity_model.pkl")
df_ranked = pd.read_csv("data/ranked_news_articles.csv")
df_proxy = pd.read_csv("data/news_with_proxy_scores.csv")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📊 Predict", "📈 Analytics", "🏆 Ranking", "🧠 Insights", "📘 About"]
)

# ======================================================
# 🏠 HOME
# ======================================================
if page == "🏠 Home":

    st.markdown("<h1 style='color:white'>📰 News Popularity Intelligence System</h1>", unsafe_allow_html=True)

    st.write("""
This system predicts the **future popularity of news articles** using AI.

It analyzes emotional tone, readability, vocabulary richness, and content depth  
to estimate how much attention a news article can attract.

Unlike traditional systems, this model works **without real labels**  
and uses intelligent proxy signals.
""")

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Articles", len(df_proxy))
    col2.metric("🔥 Avg Popularity", round(df_proxy["popularity_proxy"].mean(),2))
    col3.metric("🏆 Top Score", round(df_proxy["popularity_proxy"].max(),2))

    st.subheader("🚀 System Capabilities")

    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card">🧠 AI Prediction<br>Predicts article popularity using NLP.</div>', unsafe_allow_html=True)
    c2.markdown('<div class="card">📈 Ranking Engine<br>Ranks articles by attention score.</div>', unsafe_allow_html=True)
    c3.markdown('<div class="card">🔍 Explainable AI<br>Explains prediction logic.</div>', unsafe_allow_html=True)

    st.subheader("📊 Popularity Distribution")
    fig = px.histogram(df_proxy, x="popularity_proxy")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏆 Top 5 Trending News")
    st.dataframe(df_ranked[["title","predicted_popularity"]].head(5))


# ======================================================
# 📊 PREDICT
# ======================================================
elif page == "📊 Predict":

    st.header("📊 News Popularity Prediction Engine")

    st.write("Analyze how your article will perform using AI")

    title = st.text_input("📰 News Title")
    desc = st.text_area("📄 News Description")

    if st.button("🚀 Analyze"):

        text = title + " " + desc

        emotion = emotion_score(text)
        subjectivity = subjectivity_score(text)
        readability = readability_score(text)
        diversity = lexical_diversity(text)
        length = length_score(text)

        X = [[emotion, readability, diversity, length, subjectivity]]
        score = model.predict(X)[0]

        st.success(f"🔥 Popularity Score: {score:.2f}")

        st.subheader("📊 Feature Breakdown")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Emotion", round(emotion,2))
        c2.metric("Readability", round(readability,2))
        c3.metric("Diversity", round(diversity,2))
        c4.metric("Length", length)
        c5.metric("Subjectivity", round(subjectivity,2))

        st.subheader("📈 Feature Contribution")

        features_df = pd.DataFrame({
            "Feature":["Emotion","Readability","Diversity","Length","Subjectivity"],
            "Value":[emotion, readability, diversity, length, subjectivity]
        })

        fig = px.bar(features_df, x="Feature", y="Value", color="Value")
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Popularity Score"},
            gauge={'axis': {'range':[0,1]}}
        ))
        st.plotly_chart(fig)

        st.subheader("🧠 AI Reasoning")

        exp = explain_prediction({
            "emotion":emotion,
            "readability":readability,
            "diversity":diversity,
            "length":length
        })

        for e in exp:
            st.write("✔", e)


# ======================================================
# 📈 ANALYTICS
# ======================================================
elif page == "📈 Analytics":

    st.header("📈 Dataset Intelligence Dashboard")

    st.subheader("📊 Popularity Distribution")
    fig1 = px.histogram(df_proxy, x="popularity_proxy")
    st.plotly_chart(fig1)

    st.subheader("📊 Emotion vs Popularity")
    fig2 = px.scatter(df_proxy, x="emotion", y="popularity_proxy", color="popularity_proxy")
    st.plotly_chart(fig2)

    st.subheader("📊 Feature Correlation")

    corr = df_proxy[["emotion","readability","diversity","length","subjectivity","popularity_proxy"]].corr()
    fig3 = px.imshow(corr, text_auto=True)
    st.plotly_chart(fig3)


# ======================================================
# 🏆 RANKING
# ======================================================
elif page == "🏆 Ranking":

    st.header("🏆 News Ranking System")

    top_n = st.slider("Select Top N", 5, 50, 10)

    st.dataframe(df_ranked[["title","predicted_popularity"]].head(top_n))

    fig = px.bar(df_ranked.head(top_n),
                 x="predicted_popularity",
                 y="title",
                 orientation="h",
                 color="predicted_popularity")

    st.plotly_chart(fig, use_container_width=True)


# ======================================================
# 🧠 INSIGHTS
# ======================================================
elif page == "🧠 Insights":

    st.header("🧠 AI Insights")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Emotion", round(df_proxy["emotion"].mean(),2))
    col2.metric("Avg Readability", round(df_proxy["readability"].mean(),2))
    col3.metric("Avg Diversity", round(df_proxy["diversity"].mean(),2))

    st.subheader("🔍 Key Observations")

    st.write("""
✔ Emotional news attracts more clicks  
✔ Simple language improves reach  
✔ Longer articles increase engagement  
✔ Rich vocabulary boosts ranking  
""")

    st.subheader("📊 Top vs Low Articles")

    st.write("### 🔥 Top Articles")
    st.dataframe(df_ranked.head(5)[["title","predicted_popularity"]])

    st.write("### ⚠ Low Articles")
    st.dataframe(df_ranked.tail(5)[["title","predicted_popularity"]])


# ======================================================
# 📘 ABOUT
# ======================================================
else:

    st.header("📘 About Project")

    st.write("""
This system predicts news popularity using NLP and Machine Learning.

It uses proxy signals instead of real labels.
""")

    st.subheader("⚙️ Technologies")

    st.write("""
• Python  
• NLP  
• Machine Learning  
• Streamlit  
• Plotly  
""")

    st.subheader("🧠 Workflow")

    st.write("""
1. Text processing  
2. Feature extraction  
3. Proxy signals  
4. Model prediction  
5. Ranking  
""")