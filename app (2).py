import streamlit as st
import pandas as pd
from transformers import pipeline
import evaluate
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Dhurandhar Review Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — PROFESSIONAL DASHBOARD THEME
# ============================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* ---------- App background ---------- */
.stApp {
    background: radial-gradient(circle at top left, #171923 0%, #0B0D13 55%, #0B0D13 100%);
    color: #E6E8EC;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14161F 0%, #0E1016 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

[data-testid="stSidebar"] h1 {
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.3px;
}

/* ---------- Headings & text ---------- */
h1, h2, h3, h4, h5, h6 {
    color: #F5F6FA !important;
    font-weight: 700 !important;
}

p, label, span, div {
    color: #C7CBD4;
}

/* ---------- Hero header ---------- */
.hero {
    background: linear-gradient(135deg, rgba(229,9,20,0.16) 0%, rgba(139,10,20,0.05) 100%);
    border: 1px solid rgba(229,9,20,0.25);
    border-radius: 18px;
    padding: 2.4rem 2.6rem;
    margin-bottom: 1.6rem;
}

.hero h1 {
    font-size: 2.4rem !important;
    margin-bottom: 0.6rem;
    background: linear-gradient(90deg, #FFFFFF 0%, #FF6B6B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    font-size: 1.05rem;
    color: #B8BCC8 !important;
    max-width: 720px;
    line-height: 1.6;
}

.eyebrow {
    display: inline-block;
    background: rgba(229,9,20,0.15);
    color: #FF5C5C !important;
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 1.5px;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    margin-bottom: 1rem;
    text-transform: uppercase;
}

/* ---------- Feature / info cards ---------- */
.card {
    background: linear-gradient(180deg, #1A1D27 0%, #15171F 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.6rem 1.5rem;
    height: 100%;
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.card:hover {
    transform: translateY(-3px);
    border-color: rgba(229,9,20,0.4);
}

.card h2 {
    font-size: 1.1rem !important;
    margin-bottom: 0.4rem;
}

.card p, .card span {
    color: #9CA1AE !important;
    font-size: 0.92rem;
}

.card .icon {
    font-size: 1.8rem;
    margin-bottom: 0.6rem;
    display: block;
}

/* ---------- Section container ---------- */
.section-box {
    background: #14161F;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.8rem 1.8rem;
    margin-bottom: 1.2rem;
}

/* ---------- Buttons ---------- */
.stButton>button {
    background: linear-gradient(135deg, #E50914 0%, #B00610 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    height: 3rem;
    letter-spacing: 0.3px;
    transition: all 0.15s ease;
    box-shadow: 0 4px 14px rgba(229,9,20,0.25);
}

.stButton>button:hover {
    background: linear-gradient(135deg, #FF2A2A 0%, #E50914 100%);
    box-shadow: 0 6px 18px rgba(229,9,20,0.4);
    transform: translateY(-1px);
}

/* ---------- Text areas & inputs ---------- */
.stTextArea textarea {
    background: #1A1D27 !important;
    color: #F0F1F4 !important;
    border: 1.5px solid rgba(229,9,20,0.35) !important;
    border-radius: 12px !important;
    font-size: 0.98rem !important;
}

.stTextArea textarea:focus {
    border: 1.5px solid #E50914 !important;
    box-shadow: 0 0 0 1px rgba(229,9,20,0.3) !important;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: #1A1D27;
    border: 1.5px dashed rgba(255,255,255,0.18);
    border-radius: 12px;
}

/* ---------- Metric cards ---------- */
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, #1A1D27 0%, #15171F 100%);
    border: 1px solid rgba(255,255,255,0.07);
    padding: 1.1rem 1.2rem;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #9CA1AE !important;
    font-weight: 500;
}

div[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 700;
}

/* ---------- Dataframe ---------- */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07);
}

/* ---------- Alerts ---------- */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* ---------- Progress bar ---------- */
.stProgress > div > div {
    background: linear-gradient(90deg, #E50914, #FF5C5C);
    border-radius: 8px;
}

/* ---------- Divider ---------- */
hr {
    border-color: rgba(255,255,255,0.08) !important;
}

/* ---------- Footer ---------- */
.footer-box {
    text-align: center;
    padding: 1.6rem 0 0.6rem 0;
}

.footer-box h4 {
    color: #E50914 !important;
    font-weight: 700 !important;
    margin-bottom: 0.2rem;
}

.footer-box p {
    color: #7A7F8C !important;
    font-size: 0.85rem;
}

/* ---------- Badge pills (sidebar) ---------- */
.badge-pill {
    display: inline-block;
    width: 100%;
    background: #1A1D27;
    border: 1px solid rgba(255,255,255,0.08);
    color: #C7CBD4 !important;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 0.5rem 0.8rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("🎬 Dhurandhar Analyzer")
st.sidebar.caption("AI-powered review intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🤖 Analyze Review", "📊 Dataset Evaluation", "ℹ️ About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Built with**")
st.sidebar.markdown('<div class="badge-pill">🤗 Hugging Face Transformers</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="badge-pill">⚡ DistilBERT SST-2</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="badge-pill">🚀 Streamlit</div>', unsafe_allow_html=True)

# ============================================================
# HOME PAGE
# ============================================================
if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <span class="eyebrow">AI Sentiment Engine</span>
        <h1>Dhurandhar AI Review Analyzer</h1>
        <p>
            Analyze movie reviews instantly using Hugging Face Transformers.
            Predict whether a review is <b>Positive 😊</b> or <b>Negative 😔</b>
            in real time, powered by DistilBERT — and evaluate performance
            across entire datasets with accuracy and F1 metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown("""
        <div class="card">
            <span class="icon">🤖</span>
            <h2>AI Model</h2>
            <p>DistilBERT fine-tuned on SST-2 for high-accuracy sentiment classification.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
            <span class="icon">⚡</span>
            <h2>Fast Prediction</h2>
            <p>Real-time sentiment analysis with confidence scoring for any review.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
            <span class="icon">📈</span>
            <h2>Dataset Evaluation</h2>
            <p>Upload a CSV and instantly measure accuracy, F1 score and distribution.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.info("👈 Select a feature from the sidebar to get started.")

# ============================================================
# ANALYZE REVIEW PAGE
# ============================================================
elif page == "🤖 Analyze Review":

    st.title("🤖 Movie Review Sentiment Analysis")
    st.caption("Type or paste a review below and let the model predict its sentiment.")

    with st.container():
        st.markdown('<div class="section-box">', unsafe_allow_html=True)

        review = st.text_area(
            "Enter your review",
            height=180,
            placeholder="Example: Dhurandhar is an amazing movie with brilliant acting."
        )

        analyze_clicked = st.button("🚀 Analyze Review")
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze_clicked:

        if review.strip() == "":
            st.warning("Please enter a review before analyzing.")
        else:
            with st.spinner("Analyzing sentiment..."):
                result = classifier(review)[0]

            sentiment = result["label"]
            confidence = result["score"]

            col1, col2 = st.columns([2, 1], gap="large")

            with col1:
                if sentiment == "POSITIVE":
                    st.success("😊 Positive Review")
                else:
                    st.error("😔 Negative Review")

                st.progress(float(confidence))

                st.markdown("#### 🧠 AI Interpretation")
                if sentiment == "POSITIVE":
                    st.success("The model believes this review expresses a positive opinion.")
                else:
                    st.error("The model believes this review expresses a negative opinion.")

            with col2:
                st.metric("Confidence Score", f"{confidence*100:.2f}%")

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                number={'suffix': '%'},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "#9CA1AE"},
                    'bar': {'color': "#E50914"},
                    'bgcolor': "#1A1D27",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 50], 'color': "#22242E"},
                        {'range': [50, 100], 'color': "#2A1418"}
                    ]
                }
            ))

            gauge.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(t=30, b=10, l=30, r=30)
            )

            st.plotly_chart(gauge, use_container_width=True)

# ============================================================
# DATASET EVALUATION PAGE
# ============================================================
elif page == "📊 Dataset Evaluation":

    st.title("📊 Dataset Evaluation")
    st.caption("Upload a labeled CSV to benchmark model accuracy and F1 score.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        if "Review" not in df.columns or "Class" not in df.columns:
            st.error("CSV must contain 'Review' and 'Class' columns.")
            st.stop()

        reviews = df["Review"].astype(str).tolist()
        labels = df["Class"].tolist()

        with st.spinner("Running AI predictions..."):
            predictions = classifier(reviews)

        pred_labels = [x["label"] for x in predictions]

        if isinstance(labels[0], str):
            actual = [1 if str(x).upper() == "POSITIVE" else 0 for x in labels]
        else:
            actual = labels

        predicted = [1 if x == "POSITIVE" else 0 for x in pred_labels]

        accuracy = accuracy_metric.compute(references=actual, predictions=predicted)["accuracy"]
        f1 = f1_metric.compute(references=actual, predictions=predicted)["f1"]

        a, b, c = st.columns(3, gap="medium")
        a.metric("Accuracy", f"{accuracy*100:.2f}%")
        b.metric("F1 Score", f"{f1:.3f}")
        c.metric("Total Reviews", len(df))

        st.write("")

        result = df.copy()
        result["Prediction"] = pred_labels

        left, right = st.columns([1.4, 1], gap="large")

        with left:
            st.subheader("Prediction Results")
            st.dataframe(result, use_container_width=True, height=380)

        with right:
            st.subheader("Sentiment Distribution")
            chart = result["Prediction"].value_counts().reset_index()
            chart.columns = ["Sentiment", "Count"]

            fig = px.pie(
                chart,
                values="Count",
                names="Sentiment",
                hole=0.6,
                color="Sentiment",
                color_discrete_map={"POSITIVE": "#00C853", "NEGATIVE": "#E50914"}
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380,
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15)
            )

            st.plotly_chart(fig, use_container_width=True)

        csv = result.to_csv(index=False).encode()

        st.download_button(
            "📥 Download Predictions",
            csv,
            "Predictions.csv",
            "text/csv"
        )

# ============================================================
# ABOUT PAGE
# ============================================================
elif page == "ℹ️ About":

    st.title("🎬 About This Project")

    st.markdown("""
    <div class="section-box">

    ### Dhurandhar AI Review Analyzer

    A modern Streamlit dashboard that uses **Hugging Face DistilBERT**
    to classify movie reviews as **Positive** or **Negative**, with
    dataset-level evaluation and analytics built in.

    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div class="card">
            <h2>✨ Features</h2>
            <p>
            🤖 AI Sentiment Analysis<br>
            ⚡ Real-time Prediction<br>
            📊 Confidence Score<br>
            📈 Dataset Evaluation<br>
            📥 Download Predictions<br>
            🌙 Netflix-style Dark UI
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
            <h2>🛠 Technology Stack</h2>
            <p>
            Python<br>
            Streamlit<br>
            Hugging Face Transformers<br>
            DistilBERT SST-2<br>
            Plotly<br>
            Pandas &amp; Evaluate
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class="section-box">
        <h3>👨‍💻 Model</h3>
        <p><b>distilbert-base-uncased-finetuned-sst-2-english</b></p>
        <p>Fine-tuned on the Stanford Sentiment Treebank (SST-2).</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("Built with ❤️ using Streamlit + Hugging Face Transformers")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div class="footer-box">
    <h4>🎬 Dhurandhar AI Review Analyzer</h4>
    <p>Powered by Streamlit • Hugging Face • DistilBERT</p>
</div>
""", unsafe_allow_html=True)
