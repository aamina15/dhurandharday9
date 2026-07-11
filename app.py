import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import evaluate
import plotly.express as px
from PIL import Image
import os

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Dhurandhar AI Review Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# PREMIUM CSS
# ----------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"]{
    font-family:Poppins;
}

.stApp{
background:linear-gradient(135deg,#090909,#141414,#1d1d1d);
color:white;
}

/* Hide Streamlit Branding */
#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

/* Sidebar */

[data-testid="stSidebar"]{
background:#101010;
border-right:1px solid #303030;
}

/* Hero */

.hero{
padding:30px;
background:linear-gradient(90deg,#181818,#0b0b0b);
border-radius:20px;
box-shadow:0 0 30px rgba(229,9,20,.25);
}

.hero-title{

font-size:52px;
font-weight:800;
color:#E50914;

}

.hero-sub{

font-size:18px;
color:#DDDDDD;

}

/* Cards */

.card{

background:#181818;

padding:25px;

border-radius:18px;

border:1px solid #2c2c2c;

box-shadow:0px 8px 25px rgba(0,0,0,.4);

}

/* Button */

.stButton>button{

width:100%;

background:linear-gradient(90deg,#E50914,#ff3b3b);

color:white;

font-size:18px;

font-weight:bold;

border:none;

border-radius:12px;

padding:14px;

transition:.3s;

}

.stButton>button:hover{

transform:scale(1.02);

box-shadow:0 0 20px rgba(229,9,20,.7);

}

/* Text Area */

textarea{

background:#1b1b1b!important;

color:white!important;

border-radius:12px!important;

}

/* Metrics */

[data-testid="metric-container"]{

background:#181818;

border-radius:15px;

padding:20px;

border:1px solid #333;

}

/* Tabs */

.stTabs [data-baseweb="tab"]{

font-size:17px;

padding:12px;

}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------

@st.cache_resource
def load_model():

    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    return classifier

classifier = load_model()

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

# ----------------------------
# SIDEBAR
# ----------------------------

st.sidebar.image(
    "assets/logo.png",
    use_container_width=True
)

st.sidebar.title("🎬 Navigation")

page = st.sidebar.radio(
    "",
    [
        "🏠 Home",
        "🤖 Sentiment Analysis",
        "📊 Dataset Evaluation",
        "ℹ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Model

DistilBERT SST-2

Framework

🤗 HuggingFace

Deployment

Streamlit
"""
)

# ----------------------------
# HOME PAGE
# ----------------------------

if page=="🏠 Home":

    col1,col2=st.columns([1,2])

    with col1:

        if os.path.exists("assets/dhurandhar.jpg"):

            img=Image.open("assets/dhurandhar.jpg")

            st.image(img,use_container_width=True)

        else:

            st.image(
"https://placehold.co/500x700/111111/E50914?text=Movie+Poster",
use_container_width=True
)

    with col2:

        st.markdown("""
<div class="hero">

<div class="hero-title">

🎬 Dhurandhar AI Review Analyzer

</div>

<br>

<div class="hero-sub">

Analyze Netflix movie reviews using Hugging Face Transformers.

Predict whether a review is Positive or Negative in seconds.

Built using

✔ Streamlit

✔ Transformers

✔ DistilBERT

✔ Hugging Face

</div>

</div>
""",unsafe_allow_html=True)

    st.write("")

    c1,c2,c3=st.columns(3)

    with c1:

        st.markdown("""
<div class="card">

<h2>🤖 AI Model</h2>

DistilBERT SST-2

</div>
""",unsafe_allow_html=True)

    with c2:

        st.markdown("""
<div class="card">

<h2>⚡ Speed</h2>

Real-time Prediction

</div>
""",unsafe_allow_html=True)

    with c3:

        st.markdown("""
<div class="card">

<h2>📈 Accuracy</h2>

State-of-the-art

</div>
""",unsafe_allow_html=True)
# ==========================================
# SENTIMENT ANALYSIS PAGE
# ==========================================

elif page == "🤖 Sentiment Analysis":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            🤖 AI Sentiment Analysis
        </div>
        <br>
        <div class="hero-sub">
            Enter a Netflix movie review and let the AI determine whether it is
            Positive 😊 or Negative 😔.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    review = st.text_area(
        "📝 Enter Movie Review",
        height=220,
        placeholder="""
Example:

Dhurandhar is one of the best action thrillers I have watched.
The acting, background music and cinematography were outstanding.
        """
    )

    analyze = st.button("🚀 Analyze Review")

    if analyze:

        if review.strip() == "":
            st.warning("Please enter a review.")
            st.stop()

        with st.spinner("Analyzing review using DistilBERT..."):

            result = classifier(review)[0]

            sentiment = result["label"]
            confidence = result["score"]

        st.success("Prediction Completed!")

        st.write("")

        col1, col2 = st.columns([2,1])

        # ------------------------
        # LEFT CARD
        # ------------------------

        with col1:

            if sentiment == "POSITIVE":

                st.markdown(f"""
                <div class="card">

                <h1 style="color:#00ff99;">
                😊 POSITIVE REVIEW
                </h1>

                <hr>

                <h3>
                Audience Reaction
                </h3>

                <p style="font-size:18px;">
                The review expresses a positive opinion about the movie.
                </p>

                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="card">

                <h1 style="color:#ff4d4d;">
                😔 NEGATIVE REVIEW
                </h1>

                <hr>

                <h3>
                Audience Reaction
                </h3>

                <p style="font-size:18px;">
                The review expresses a negative opinion about the movie.
                </p>

                </div>
                """, unsafe_allow_html=True)

        # ------------------------
        # RIGHT CARD
        # ------------------------

        with col2:

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            st.progress(float(confidence))

        st.write("")

        # ------------------------
        # Plotly Gauge
        # ------------------------

        import plotly.graph_objects as go

        gauge = go.Figure(go.Indicator(

            mode="gauge+number",

            value=confidence*100,

            number={'suffix':"%"},

            title={'text':"Prediction Confidence"},

            gauge={

                'axis':{'range':[0,100]},

                'bar':{'color':'red'},

                'steps':[

                    {'range':[0,40],'color':'#3a3a3a'},

                    {'range':[40,70],'color':'#666666'},

                    {'range':[70,100],'color':'#00ff99'}

                ]

            }

        ))

        gauge.update_layout(

            height=350,

            paper_bgcolor="#111111",

            font=dict(color="white",size=18)

        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        st.write("")

        # ------------------------
        # AI Summary Card
        # ------------------------

        st.markdown("""
        <div class="card">

        <h2>🤖 AI Interpretation</h2>

        The model analyzed the semantic meaning of the review using
        DistilBERT (Fine-tuned on SST-2).

        It predicts whether the review is Positive or Negative based
        on contextual understanding rather than simple keywords.

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # ------------------------
        # Probability Chart
        # ------------------------

        if sentiment == "POSITIVE":

            chart = pd.DataFrame({

                "Sentiment":[
                    "Positive",
                    "Negative"
                ],

                "Probability":[
                    confidence,
                    1-confidence
                ]

            })

        else:

            chart = pd.DataFrame({

                "Sentiment":[
                    "Negative",
                    "Positive"
                ],

                "Probability":[
                    confidence,
                    1-confidence
                ]

            })

        fig = px.bar(

            chart,

            x="Sentiment",

            y="Probability",

            text="Probability",

            color="Sentiment",

            color_discrete_sequence=["#E50914","#00C853"]

        )

        fig.update_layout(

            template="plotly_dark",

            height=400,

            paper_bgcolor="#111111",

            plot_bgcolor="#111111"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        st.write("")

        st.download_button(

            "📥 Download Prediction",

            data=f"""
Review:

{review}

Prediction:

{sentiment}

Confidence:

{confidence*100:.2f}%
""",

            file_name="prediction.txt"

        )
# ==========================================
# DATASET EVALUATION
# ==========================================

elif page == "📊 Dataset Evaluation":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            📊 Dataset Evaluation Dashboard
        </div>

        <br>

        <div class="hero-sub">
            Upload your dataset and evaluate the DistilBERT sentiment model.
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    uploaded_file = st.file_uploader(
        "📂 Upload CSV Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.success("Dataset Uploaded Successfully!")

        st.write("")

        st.subheader("Dataset Preview")

        st.dataframe(df.head(), use_container_width=True)

        st.write("")

        if "Review" not in df.columns or "Class" not in df.columns:

            st.error(
                "Dataset must contain 'Review' and 'Class' columns."
            )

            st.stop()

        reviews = df["Review"].astype(str).tolist()

        labels = df["Class"].tolist()

        with st.spinner("Running predictions on dataset..."):

            predictions = classifier(reviews)

        predicted = [item["label"] for item in predictions]

        # --------------------------
        # Convert labels
        # --------------------------

        if isinstance(labels[0], str):

            true_labels = [

                1 if str(x).upper() == "POSITIVE"

                else 0

                for x in labels

            ]

        else:

            true_labels = labels

        pred_labels = [

            1 if x == "POSITIVE"

            else 0

            for x in predicted

        ]

        accuracy = accuracy_metric.compute(

            references=true_labels,

            predictions=pred_labels

        )["accuracy"]

        f1 = f1_metric.compute(

            references=true_labels,

            predictions=pred_labels

        )["f1"]

        st.write("")

        # --------------------------
        # Metrics
        # --------------------------

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Accuracy", f"{accuracy*100:.2f}%")

        with c2:
            st.metric("F1 Score", f"{f1:.3f}")

        with c3:
            st.metric("Reviews", len(df))

        st.write("")

        # --------------------------
        # Result DataFrame
        # --------------------------

        result_df = df.copy()

        result_df["Prediction"] = predicted

        st.subheader("Prediction Results")

        st.dataframe(

            result_df,

            use_container_width=True

        )

        st.write("")

        # --------------------------
        # Pie Chart
        # --------------------------

        sentiment_count = pd.DataFrame(

            result_df["Prediction"].value_counts()

        ).reset_index()

        sentiment_count.columns = [

            "Sentiment",

            "Count"

        ]

        pie = px.pie(

            sentiment_count,

            values="Count",

            names="Sentiment",

            hole=.55,

            color="Sentiment",

            color_discrete_map={

                "POSITIVE": "#00C853",

                "NEGATIVE": "#E50914"

            }

        )

        pie.update_layout(

            template="plotly_dark",

            paper_bgcolor="#111111",

            plot_bgcolor="#111111",

            height=450

        )

        st.plotly_chart(

            pie,

            use_container_width=True

        )

        # --------------------------
        # Bar Chart
        # --------------------------

        bar = px.bar(

            sentiment_count,

            x="Sentiment",

            y="Count",

            text="Count",

            color="Sentiment",

            color_discrete_map={

                "POSITIVE": "#00C853",

                "NEGATIVE": "#E50914"

            }

        )

        bar.update_layout(

            template="plotly_dark",

            height=450,

            paper_bgcolor="#111111",

            plot_bgcolor="#111111"

        )

        st.plotly_chart(

            bar,

            use_container_width=True

        )

        # --------------------------
        # Download CSV
        # --------------------------

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(

            "📥 Download Predictions",

            csv,

            file_name="Predicted_Reviews.csv",

            mime="text/csv"

        )

        st.write("")

        st.success("Evaluation Completed Successfully!")
        # ==========================================
# ABOUT PAGE
# ==========================================

elif page == "ℹ About":

    st.markdown("""
    <div class="hero">

    <div class="hero-title">
    🎬 About This Project
    </div>

    <br>

    <div class="hero-sub">

    AI Powered Netflix Movie Review Sentiment Analyzer

    </div>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("""
        <div class="card">

        <h2>📖 Project Overview</h2>

        <hr>

        <p style="font-size:18px">

        This application predicts whether a movie review is
        <b>Positive</b> or <b>Negative</b> using a pretrained
        Hugging Face Transformer model.

        Users can:

        ✔ Analyze individual reviews

        ✔ Upload datasets

        ✔ Evaluate Accuracy & F1 Score

        ✔ Visualize Predictions

        ✔ Download prediction results

        </p>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="card">

        <h2>🤖 AI Model</h2>

        <hr>

        <p style="font-size:18px">

        Model

        <b>DistilBERT SST-2</b>

        <br><br>

        Framework

        Hugging Face Transformers

        <br><br>

        Task

        Binary Sentiment Classification

        <br><br>

        Optimized for fast inference while maintaining
        excellent accuracy.

        </p>

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.subheader("🛠 Technology Stack")

    a, b, c, d = st.columns(4)

    with a:
        st.metric("Frontend", "Streamlit")

    with b:
        st.metric("Model", "DistilBERT")

    with c:
        st.metric("Library", "Transformers")

    with d:
        st.metric("Language", "Python")

    st.write("")

    st.subheader("📈 Features")

    feature1, feature2 = st.columns(2)

    with feature1:

        st.success("✔ Real-time Sentiment Analysis")

        st.success("✔ Confidence Score")

        st.success("✔ Interactive Charts")

        st.success("✔ Dataset Evaluation")

    with feature2:

        st.success("✔ Accuracy & F1 Score")

        st.success("✔ CSV Upload")

        st.success("✔ Download Predictions")

        st.success("✔ Responsive UI")

    st.write("")

    st.markdown("""
    ---
    <center>

    <h3 style="color:#E50914;">

    🎬 Dhurandhar AI Review Analyzer

    </h3>

    <p>

    Developed using ❤️ Streamlit + Hugging Face Transformers

    </p>

    </center>
    """, unsafe_allow_html=True)

