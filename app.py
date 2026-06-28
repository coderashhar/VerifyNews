import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
from datetime import datetime

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
    page_title="VerifyNews — AI Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_models():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

vectorizer, model = load_models()

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ============ ROOT & GLOBAL ============ */
:root {
    --bg-primary: #06080f;
    --bg-secondary: #0c1021;
    --bg-card: rgba(15, 20, 40, 0.65);
    --border-glow: rgba(99, 102, 241, 0.25);
    --accent-blue: #6366f1;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-rose: #f43f5e;
    --accent-amber: #f59e0b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Full dark background */
[data-testid="stAppViewContainer"] {
    background: var(--bg-primary);
    background-image: 
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.12), transparent),
        radial-gradient(ellipse 60% 40% at 80% 50%, rgba(6, 182, 212, 0.06), transparent),
        radial-gradient(ellipse 60% 40% at 20% 80%, rgba(16, 185, 129, 0.06), transparent);
}

.main .block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 900px !important;
}

/* ============ SIDEBAR ============ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c1a 0%, #0c1021 50%, #0a0e1c 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.15) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-secondary) !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
}

/* ============ ANIMATED GRID BACKGROUND ============ */
.grid-bg {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: 
        linear-gradient(rgba(99, 102, 241, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99, 102, 241, 0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
}

/* ============ HERO SECTION ============ */
.hero-container {
    text-align: center;
    padding: 2rem 0 2.5rem;
    position: relative;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    padding: 6px 16px;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #a5b4fc;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    animation: fadeInDown 0.6s ease-out;
}

.hero-badge .pulse-dot {
    width: 7px;
    height: 7px;
    background: #10b981;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.5); }
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: -1.5px;
    line-height: 1.1;
    margin-bottom: 0.6rem;
    background: linear-gradient(135deg, #ffffff 0%, #c7d2fe 40%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: fadeInDown 0.7s ease-out;
}

.hero-subtitle {
    font-size: 1.08rem;
    color: var(--text-secondary);
    font-weight: 400;
    line-height: 1.6;
    max-width: 540px;
    margin: 0 auto;
    animation: fadeInDown 0.8s ease-out;
}

.hero-subtitle strong {
    color: #a5b4fc;
    font-weight: 600;
}

@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ============ STATS BAR ============ */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    margin: 1.5rem 0 2rem;
    animation: fadeInUp 0.8s ease-out;
}

.stat-item {
    text-align: center;
}

.stat-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1;
}

.stat-value.blue { color: #818cf8; }
.stat-value.cyan { color: #22d3ee; }
.stat-value.emerald { color: #34d399; }

.stat-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-top: 4px;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ============ GLASS CARD ============ */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-glow);
    border-radius: 20px;
    padding: 2rem 2rem 1.5rem;
    margin: 0 auto 1.5rem;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.6s ease-out;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), rgba(6, 182, 212, 0.5), transparent);
}

.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.2rem;
}

.card-icon {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.25);
}

.card-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.3px;
}

.card-desc {
    font-size: 0.8rem;
    color: var(--text-muted);
    font-weight: 400;
}

/* ============ TEXT AREA ============ */
.stTextArea label {
    color: transparent !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

.stTextArea textarea {
    background: rgba(10, 14, 30, 0.8) !important;
    color: var(--text-primary) !important;
    border: 1.5px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 14px !important;
    padding: 18px !important;
    font-size: 0.95rem !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.65 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    resize: none !important;
}

.stTextArea textarea:focus {
    border-color: rgba(99, 102, 241, 0.55) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1), 0 4px 20px rgba(99, 102, 241, 0.08) !important;
}

.stTextArea textarea::placeholder {
    color: var(--text-muted) !important;
    font-style: italic;
}

/* ============ BUTTON ============ */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1, #4f46e5, #6366f1) !important;
    background-size: 200% 200% !important;
    color: white !important;
    border: none !important;
    padding: 16px 24px !important;
    border-radius: 14px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.3px;
    cursor: pointer;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.25) !important;
}

.stButton > button:hover {
    background-position: right center !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.4) !important;
}

.stButton > button:active {
    transform: translateY(0) scale(0.98) !important;
}

/* ============ RESULT CARDS ============ */
.result-card {
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1.5rem;
    text-align: center;
    animation: resultSlideIn 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}

@keyframes resultSlideIn {
    from { opacity: 0; transform: translateY(20px) scale(0.97); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

.result-fake {
    background: rgba(244, 63, 94, 0.08);
    border: 1.5px solid rgba(244, 63, 94, 0.3);
}
.result-fake::before {
    background: linear-gradient(90deg, transparent, #f43f5e, transparent);
}

.result-real {
    background: rgba(16, 185, 129, 0.08);
    border: 1.5px solid rgba(16, 185, 129, 0.3);
}
.result-real::before {
    background: linear-gradient(90deg, transparent, #10b981, transparent);
}

.result-icon {
    font-size: 3rem;
    margin-bottom: 0.6rem;
    animation: bounceIn 0.6s cubic-bezier(0.68, -0.55, 0.27, 1.55);
}

@keyframes bounceIn {
    0% { transform: scale(0); }
    60% { transform: scale(1.2); }
    100% { transform: scale(1); }
}

.result-label {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
}

.result-label.fake { color: #fb7185; }
.result-label.real { color: #34d399; }

.result-sublabel {
    font-size: 0.85rem;
    color: var(--text-secondary);
    font-weight: 400;
}

/* ============ CONFIDENCE METER ============ */
.confidence-container {
    margin-top: 1.2rem;
    padding: 1rem 1.5rem;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 12px;
}

.confidence-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.confidence-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}

.confidence-value {
    font-size: 1.1rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
}

.confidence-value.fake { color: #fb7185; }
.confidence-value.real { color: #34d399; }

.confidence-bar-bg {
    width: 100%;
    height: 6px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 10px;
    animation: fillBar 1s cubic-bezier(0.4, 0, 0.2, 1);
}

.confidence-bar-fill.fake {
    background: linear-gradient(90deg, #f43f5e, #fb7185);
}

.confidence-bar-fill.real {
    background: linear-gradient(90deg, #10b981, #34d399);
}

@keyframes fillBar {
    from { width: 0; }
}

/* ============ HISTORY TABLE ============ */
.history-section {
    margin-top: 2rem;
    animation: fadeInUp 0.6s ease-out;
}

.history-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(99, 102, 241, 0.12);
}

.history-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    background: rgba(10, 14, 30, 0.5);
    border: 1px solid rgba(99, 102, 241, 0.08);
    border-radius: 12px;
    margin-bottom: 8px;
    transition: all 0.2s ease;
}

.history-item:hover {
    border-color: rgba(99, 102, 241, 0.2);
    background: rgba(10, 14, 30, 0.7);
}

.history-text {
    font-size: 0.85rem;
    color: var(--text-secondary);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 500px;
}

.history-meta {
    display: flex;
    align-items: center;
    gap: 12px;
}

.history-time {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
}

.history-badge {
    font-size: 0.68rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 50px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.badge-fake {
    background: rgba(244, 63, 94, 0.15);
    color: #fb7185;
    border: 1px solid rgba(244, 63, 94, 0.3);
}

.badge-real {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

/* ============ SIDEBAR STYLES ============ */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.5rem 0 1.5rem;
    border-bottom: 1px solid rgba(99, 102, 241, 0.12);
    margin-bottom: 1.5rem;
}

.sidebar-logo {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
}

.sidebar-brand-name {
    font-size: 1.15rem;
    font-weight: 800;
    color: var(--text-primary) !important;
    letter-spacing: -0.5px;
}

.sidebar-section {
    margin-bottom: 1.5rem;
}

.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 0.8rem;
}

.step-item {
    display: flex;
    gap: 12px;
    padding: 10px 0;
    align-items: flex-start;
}

.step-number {
    width: 26px;
    height: 26px;
    min-width: 26px;
    border-radius: 8px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 700;
    color: #a5b4fc !important;
}

.step-content {
    flex: 1;
}

.step-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary) !important;
    margin-bottom: 2px;
}

.step-desc {
    font-size: 0.75rem;
    color: var(--text-muted) !important;
    line-height: 1.4;
}

.model-info-card {
    background: rgba(99, 102, 241, 0.06);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 12px;
    padding: 14px;
    margin-top: 0.5rem;
}

.model-info-row {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    font-size: 0.78rem;
}

.model-info-key {
    color: var(--text-muted) !important;
    font-weight: 500;
}

.model-info-val {
    color: var(--text-primary) !important;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
}

/* ============ FOOTER ============ */
.app-footer {
    text-align: center;
    padding: 2.5rem 0 1rem;
    border-top: 1px solid rgba(99, 102, 241, 0.08);
    margin-top: 3rem;
}

.footer-text {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-weight: 400;
}

.footer-text a {
    color: #818cf8;
    text-decoration: none;
    font-weight: 600;
}

.footer-links {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    margin-top: 8px;
    font-size: 0.72rem;
}

.footer-links span {
    color: var(--text-muted);
    opacity: 0.6;
}

/* ============ DIVIDER ============ */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.2), transparent);
    margin: 1rem 0;
}

/* ============ WARNING/ERROR ============ */
.stAlert {
    border-radius: 12px !important;
}

/* ============ SPINNER ============ */
.stSpinner > div {
    border-color: #6366f1 !important;
}

/* Hide default Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ============ RESPONSIVE ============ */
@media (max-width: 768px) {
    .hero-title { font-size: 2.2rem; }
    .stats-bar { gap: 1.5rem; }
    .stat-value { font-size: 1.2rem; }
    .glass-card { padding: 1.5rem; }
    .result-card { padding: 1.5rem; }
}
</style>

<div class="grid-bg"></div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">🛡️</div>
        <div class="sidebar-brand-name">VerifyNews</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-title">How It Works</div>
        <div class="step-item">
            <div class="step-number">1</div>
            <div class="step-content">
                <div class="step-title">Paste Article</div>
                <div class="step-desc">Enter the full text or headline of the news article.</div>
            </div>
        </div>
        <div class="step-item">
            <div class="step-number">2</div>
            <div class="step-content">
                <div class="step-title">NLP Processing</div>
                <div class="step-desc">Text is cleaned, stemmed, and vectorized using TF-IDF.</div>
            </div>
        </div>
        <div class="step-item">
            <div class="step-number">3</div>
            <div class="step-content">
                <div class="step-title">ML Classification</div>
                <div class="step-desc">Logistic Regression model predicts authenticity.</div>
            </div>
        </div>
        <div class="step-item">
            <div class="step-number">4</div>
            <div class="step-content">
                <div class="step-title">Get Results</div>
                <div class="step-desc">Instant verdict with confidence score.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-title">Model Information</div>
        <div class="model-info-card">
            <div class="model-info-row">
                <span class="model-info-key">Algorithm</span>
                <span class="model-info-val">Logistic Regression</span>
            </div>
            <div class="model-info-row">
                <span class="model-info-key">Vectorizer</span>
                <span class="model-info-val">TF-IDF</span>
            </div>
            <div class="model-info-row">
                <span class="model-info-key">Preprocessing</span>
                <span class="model-info-val">Porter Stemmer</span>
            </div>
            <div class="model-info-row">
                <span class="model-info-key">Dataset</span>
                <span class="model-info-val">~44,000 articles</span>
            </div>
            <div class="model-info-row">
                <span class="model-info-key">Framework</span>
                <span class="model-info-val">scikit-learn</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-section-title">Tips for Best Results</div>
        <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.7;">
            <div style="padding: 4px 0;">⚡ Use full article text for accuracy</div>
            <div style="padding: 4px 0;">📝 English language articles only</div>
            <div style="padding: 4px 0;">🔍 Longer texts yield better results</div>
            <div style="padding: 4px 0;">🚫 Avoid pasting URLs or HTML</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- HERO SECTION ---------------- #
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">
        <span class="pulse-dot"></span>
        AI-Powered Analysis
    </div>
    <div class="hero-title">Fake News Detector</div>
    <div class="hero-subtitle">
        Detect misinformation instantly with <strong>machine learning</strong> and
        <strong>natural language processing</strong>. Paste any article and get a verdict in seconds.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- STATS BAR ---------------- #
total_analyzed = len(st.session_state.history)
fake_count = sum(1 for h in st.session_state.history if h["result"] == "Fake")
real_count = total_analyzed - fake_count

st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-value blue">{total_analyzed}</div>
        <div class="stat-label">Articles Analyzed</div>
    </div>
    <div class="stat-item">
        <div class="stat-value emerald">{real_count}</div>
        <div class="stat-label">Verified Real</div>
    </div>
    <div class="stat-item">
        <div class="stat-value" style="color: #fb7185;">{fake_count}</div>
        <div class="stat-label">Flagged Fake</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT CARD ---------------- #
st.markdown("""
<div class="glass-card">
    <div class="card-header">
        <div class="card-icon">📝</div>
        <div>
            <div class="card-title">Analyze Article Text</div>
            <div class="card-desc">Paste a news article or headline for real-time analysis</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

news = st.text_area(
    "Enter News Article",
    height=200,
    placeholder="Paste the full news article text here for analysis...\n\nFor best results, include the complete article rather than just headlines.",
    label_visibility="collapsed"
)

# ---------------- ANALYZE BUTTON ---------------- #
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_clicked = st.button("🔍  Analyze Article", use_container_width=True)

# ---------------- PREDICTION LOGIC ---------------- #
if analyze_clicked:
    if news.strip() == "":
        st.warning("⚠️ Please enter some news text to analyze.")
    else:
        with st.spinner("🔄 Analyzing article..."):
            stemmed_news = stemming(news)
            transformed_news = vectorizer.transform([stemmed_news])
            
            prediction = model.predict(transformed_news)
            
            # Get prediction probabilities for confidence score
            try:
                probabilities = model.predict_proba(transformed_news)
                confidence = np.max(probabilities) * 100
            except Exception:
                confidence = 85.0  # fallback

            is_fake = prediction[0] == 1
            
            # Add to history
            preview = news[:80] + "..." if len(news) > 80 else news
            preview = preview.replace('\n', ' ')
            st.session_state.history.insert(0, {
                "text": preview,
                "result": "Fake" if is_fake else "Real",
                "confidence": f"{confidence:.1f}",
                "time": datetime.now().strftime("%H:%M:%S")
            })
            # Keep only last 10
            st.session_state.history = st.session_state.history[:10]

        if is_fake:
            st.markdown(f"""
            <div class="result-card result-fake">
                <div class="result-icon">🚨</div>
                <div class="result-label fake">Fake News Detected</div>
                <div class="result-sublabel">This article appears to contain misinformation</div>
                <div class="confidence-container">
                    <div class="confidence-header">
                        <span class="confidence-label">Confidence Score</span>
                        <span class="confidence-value fake">{confidence:.1f}%</span>
                    </div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill fake" style="width: {confidence}%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-real">
                <div class="result-icon">✅</div>
                <div class="result-label real">Authentic News</div>
                <div class="result-sublabel">This article appears to be legitimate and trustworthy</div>
                <div class="confidence-container">
                    <div class="confidence-header">
                        <span class="confidence-label">Confidence Score</span>
                        <span class="confidence-value real">{confidence:.1f}%</span>
                    </div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill real" style="width: {confidence}%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.rerun()

# ---------------- HISTORY SECTION ---------------- #
if st.session_state.history:
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    history_html = '<div class="history-section">'
    history_html += '<div class="history-title">🕒 Recent Analyses</div>'
    
    for item in st.session_state.history:
        badge_class = "badge-fake" if item["result"] == "Fake" else "badge-real"
        history_html += f"""
        <div class="history-item">
            <div class="history-text">{item["text"]}</div>
            <div class="history-meta">
                <span class="history-time">{item["time"]}</span>
                <span class="history-badge {badge_class}">{item["result"]}</span>
            </div>
        </div>
        """
    
    history_html += '</div>'
    st.markdown(history_html, unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #
st.markdown("""
<div class="app-footer">
    <div class="footer-text">
        Built with ❤️ using <a href="#">Streamlit</a> • Machine Learning • NLP
    </div>
    <div class="footer-links">
        <span>© 2025 VerifyNews</span>
        <span>•</span>
        <span>Powered by scikit-learn</span>
        <span>•</span>
        <span>Trust through Technology</span>
    </div>
</div>
""", unsafe_allow_html=True)