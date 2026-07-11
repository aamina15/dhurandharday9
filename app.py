import streamlit as st
import pandas as pd
from transformers import pipeline
import evaluate
import plotly.express as px
import plotly.graph_objects as go

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="🎬 Dhurandhar Review Analyzer",
    page_icon="🎬",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>

.stApp{
background:#0E1117;
color:white;
}

[data-testid="stSidebar"]{
background:#161B22;
}

h1,h2,h3,h4,h5,h6,p,label{
color:white !important;
}

.stButton>button{
background:#E50914;
color:white;
border:none;
border-radius:10px;
font-weight:bold;
height:3rem;
}

.stButton>button:hover{
background:#ff2a2a;
}

.stTextArea textarea{
background:#1F2937 !important;
color:white !important;
border:2px solid #E50914 !important;
border-radius:10px;
}

.stFileUploader{
background:#1F2937;
padding:10px;
border-radius:10px;
}

div[data-testid="metric-container"]{
background:#1F2937;
padding:15px;
border-radius:10px;
}

</style>
""", unsafe_allow_html=True)
# -------------------- MODEL --------------------

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

# -------------------- SIDEBAR --------------------

st.sidebar.title("🎬 Navigation")

page = st.sidebar.radio(
    "",
    [
        "🏠 Home",
        "🤖 Analyze Review",
        "📊 Dataset Evaluation",
        "ℹ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("🤗 Hugging Face")
st.sidebar.success("⚡ DistilBERT")
st.sidebar.success("🚀 Streamlit")

# -------------------- HOME --------------------

if page=="🏠 Home":

    st.markdown("""
    <div class="hero">

    <h1>🎬 Dhurandhar AI Review Analyzer</h1>

    <p>

    Analyze movie reviews using Hugging Face Transformers.

    Predict whether a review is Positive 😊 or Negative 😔

    instantly using DistilBERT.

    </p>

    </div>

    """, unsafe_allow_html=True)

    st.write("")

    c1,c2,c3=st.columns(3)

    with c1:

        st.markdown("""
        <div class="card">

        <h2>🤖 AI Model</h2>

        DistilBERT SST-2

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="card">

        <h2>⚡ Fast Prediction</h2>

        Real-time Sentiment Analysis

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown("""
        <div class="card">

        <h2>📈 Dataset Evaluation</h2>

        Accuracy & F1 Score

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.info("👈 Select a feature from the sidebar to get started.")

# -------------------- SENTIMENT ANALYSIS --------------------

elif page=="🤖 Analyze Review":

    st.title("🤖 Movie Review Sentiment Analysis")

    review = st.text_area(

        "Enter your review",

        height=200,

        placeholder="Example: Dhurandhar is an amazing movie with brilliant acting."

    )

    if st.button("🚀 Analyze"):

        if review.strip()=="":

            st.warning("Please enter a review.")

        else:

            with st.spinner("Analyzing..."):

                result=classifier(review)[0]

            sentiment=result["label"]

            confidence=result["score"]

            col1,col2=st.columns([2,1])

            with col1:

                if sentiment=="POSITIVE":

                    st.success("😊 Positive Review")

                else:

                    st.error("😔 Negative Review")

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence*100:.2f}%"
                )

            st.progress(float(confidence))

            gauge = go.Figure(go.Indicator(

                mode="gauge+number",

                value=confidence*100,

                number={'suffix':'%'},

                gauge={

                    'axis':{'range':[0,100]},

                    'bar':{'color':'red'}

                }

            ))

            gauge.update_layout(

                template="plotly_dark",

                height=320

            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

            st.markdown("### 🤖 AI Interpretation")

            if sentiment=="POSITIVE":

                st.success(
                    "The model believes this review expresses a positive opinion."
                )

            else:

                st.error(
                    "The model believes this review expresses a negative opinion."
                )
            # ==========================
            # DATASET EVALUATION
            # ==========================

elif page == "📊 Dataset Evaluation":

    st.title("📊 Dataset Evaluation")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

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

        accuracy = accuracy_metric.compute(
            references=actual,
            predictions=predicted
        )["accuracy"]

        f1 = f1_metric.compute(
            references=actual,
            predictions=predicted
        )["f1"]

        a, b, c = st.columns(3)

        a.metric("Accuracy", f"{accuracy*100:.2f}%")
        b.metric("F1 Score", f"{f1:.3f}")
        c.metric("Reviews", len(df))

        result = df.copy()
        result["Prediction"] = pred_labels

        st.subheader("Prediction Results")
        st.dataframe(result, use_container_width=True)

        chart = result["Prediction"].value_counts().reset_index()
        chart.columns = ["Sentiment", "Count"]

        fig = px.pie(
            chart,
            values="Count",
            names="Sentiment",
            hole=0.55,
            color="Sentiment",
            color_discrete_map={
                "POSITIVE": "#00C853",
                "NEGATIVE": "#E50914"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)

        csv = result.to_csv(index=False).encode()

        st.download_button(
            "📥 Download Predictions",
            csv,
            "Predictions.csv",
            "text/csv"
        )

# ==========================
# ABOUT
# ==========================

elif page == "ℹ About":

    st.title("🎬 About")

    st.markdown("""
### Dhurandhar AI Review Analyzer

A modern Streamlit web application that uses **Hugging Face DistilBERT**
to classify movie reviews as **Positive** or **Negative**.

---

### ✨ Features

- 🤖 AI Sentiment Analysis
- ⚡ Real-time Prediction
- 📊 Confidence Score
- 📈 Dataset Evaluation
- 📥 Download Predictions
- 🌙 Netflix-style Dark UI

---

### 🛠 Technology Stack

- Python
- Streamlit
- Hugging Face Transformers
- DistilBERT SST-2
- Plotly
- Pandas
- Evaluate

---

### 👨‍💻 Model

**distilbert-base-uncased-finetuned-sst-2-english**

Fine-tuned on the Stanford Sentiment Treebank (SST-2).

""")

    st.info("Built using ❤️ Streamlit + Hugging Face Transformers")

# ==========================
# FOOTER
# ==========================

st.markdown("---")

st.markdown(
"""
<center>

<h4 style='color:#E50914;'>
🎬 Dhurandhar AI Review Analyzer
</h4>

<p>
Powered by Streamlit • Hugging Face • DistilBERT
</p>

</center>
""",
unsafe_allow_html=True
)                

