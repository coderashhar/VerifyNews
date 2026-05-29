import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords', quiet=True)
ps = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)  
    stemmed_content = stemmed_content.lower() 
    stemmed_content = stemmed_content.split() 
    stemmed_content = [
        ps.stem(word) 
        for word in stemmed_content 
        if word not in stopwords.words('english')
    ]
    stemmed_content = ' '.join(stemmed_content)    
    return stemmed_content

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ---------------- LOAD MODEL ---------------- #
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.title {
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    margin-top: 20px;
    color: #ffffff;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 40px;
    font-size: 1.1rem;
}

.stTextArea textarea {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 15px !important;
    border: 2px solid #334155 !important;
    padding: 15px !important;
    font-size: 16px !important;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    padding: 14px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0px 0px 20px rgba(124, 58, 237, 0.6);
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 25px;
}

.fake {
    background-color: rgba(239,68,68,0.2);
    color: #ef4444;
    border: 2px solid #ef4444;
}

.real {
    background-color: rgba(34,197,94,0.2);
    color: #22c55e;
    border: 2px solid #22c55e;
}

.footer {
    text-align: center;
    margin-top: 40px;
    color: #94a3b8;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.markdown("<div class='title'>📰 Fake News Detector</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>AI-powered Fake News Detection using Machine Learning & NLP</div>",
    unsafe_allow_html=True
)

# ---------------- INPUT ---------------- #
news = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste the news article here..."
)

# ---------------- BUTTON ---------------- #
if st.button("Analyze News"):

    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:

        stemmed_news = stemming(news)
        transformed_news = vectorizer.transform([stemmed_news])

        prediction = model.predict(transformed_news)

        if prediction[0] == 1:
            st.markdown(
                "<div class='result-box fake'>🚨 Fake News Detected</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-box real'>✅ Real News Detected</div>",
                unsafe_allow_html=True
            )

# ---------------- FOOTER ---------------- #
st.markdown(
    "<div class='footer'>Built with Streamlit • Machine Learning • NLP</div>",
    unsafe_allow_html=True
)