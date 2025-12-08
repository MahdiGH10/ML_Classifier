import streamlit as st
import joblib
import json
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    except:
        pass

download_nltk_data()

# Page configuration
st.set_page_config(
    page_title="News Article Classifier | ITBS",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with Animations and Glassmorphism
st.markdown("""
<style>
    /* Import Professional Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    html {
        scroll-behavior: smooth;
    }
    
    /* Animated gradient background */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main {
        background-image: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==');
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(102, 74, 162, 0.85) 50%, rgba(15, 23, 42, 0.95) 100%), 
                    url('../assets/background.png');
        background-size: 200% 200%, cover;
        background-position: 0% 50%, center;
        background-attachment: fixed, fixed;
        background-repeat: no-repeat, no-repeat;
        animation: gradientShift 15s ease infinite;
    }
    
    @supports not (backdrop-filter: blur(10px)) {
        .main {
            background: linear-gradient(135deg, #0f172a 0%, #664aa2 50%, #0f172a 100%);
            background-size: 200% 200%;
            animation: gradientShift 15s ease infinite;
        }
    }
    
    /* Header Animation */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.3);
        animation: fadeInDown 0.8s ease-out;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Card Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Glassmorphism Card */
    .card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 1.5rem;
        animation: fadeInUp 0.6s ease-out;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
        background: rgba(255, 255, 255, 1);
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Prediction Result Animation */
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.8);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.4);
        animation: scaleIn 0.5s ease-out;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .prediction-result:hover {
        animation: pulse 2s ease-in-out infinite;
    }
    
    .prediction-category {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    
    .prediction-confidence {
        font-size: 1.2rem;
        font-weight: 500;
        opacity: 0.95;
    }
    
    /* Metric Card Animation */
    @keyframes countUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        animation: countUp 0.6s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px) rotate(2deg);
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.25);
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Button Animation */
    @keyframes shine {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        background-size: 200% auto;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.875rem 2rem;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6);
        animation: shine 1.5s ease-in-out;
    }
    
    .stButton>button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    /* Sidebar Glassmorphism */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-right: 1px solid rgba(226, 232, 240, 0.5);
    }
    
    /* Info Box Animation */
    .info-box {
        background: linear-gradient(135deg, rgba(240, 249, 255, 0.9) 0%, rgba(224, 242, 254, 0.9) 100%);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-left: 4px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .info-box:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.15);
        border-left-width: 6px;
    }
    
    .info-box-title {
        font-weight: 600;
        color: #1e40af;
        margin-bottom: 0.5rem;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-item {
        background: rgba(248, 250, 252, 0.9);
        backdrop-filter: blur(5px);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stat-item:hover {
        background: rgba(255, 255, 255, 1);
        transform: scale(1.05);
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 0.25rem;
    }
    
    /* Text Input with smooth transitions */
    .stTextArea textarea {
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        font-size: 1rem;
        padding: 1rem;
        font-family: 'Inter', sans-serif;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(5px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.15);
        background: white;
        transform: scale(1.01);
    }
    
    .stTextArea textarea:hover {
        border-color: #93c5fd;
    }
    
    /* Tab Animation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.95);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        transform: scale(1.05);
    }
    
    /* Category Badges with animations */
    .category-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.25rem;
        transition: all 0.3s ease;
        cursor: default;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .category-badge:hover {
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .badge-business { 
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
        color: #1e40af; 
    }
    .badge-sports { 
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
        color: #166534; 
    }
    .badge-tech { 
        background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%); 
        color: #6b21a8; 
    }
    .badge-world { 
        background: linear-gradient(135deg, #fef9c3 0%, #fef08a 100%); 
        color: #854d0e; 
    }
    
    /* Sample Article Cards */
    .sample-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border-left: 4px solid;
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .sample-card:hover {
        transform: translateX(8px) scale(1.02);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        border-left-width: 6px;
    }
    
    .sample-card.business { 
        border-left-color: #3b82f6;
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(219,234,254,0.3) 100%);
    }
    .sample-card.sports { 
        border-left-color: #22c55e;
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(220,252,231,0.3) 100%);
    }
    .sample-card.tech { 
        border-left-color: #a855f7;
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(243,232,255,0.3) 100%);
    }
    .sample-card.world { 
        border-left-color: #eab308;
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(254,249,195,0.3) 100%);
    }
    
    /* Loading Animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Fade in animation */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Selection color */
    ::selection {
        background: rgba(59, 130, 246, 0.3);
        color: #1e3a8a;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: white;
        margin-top: 3rem;
        animation: fadeIn 1s ease-in;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_models():
    import os
    # Get the project root directory (parent of src/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    models_path = os.path.join(project_root, 'models', 'FinalModel', 'models_4class')
    
    model = joblib.load(os.path.join(models_path, 'best_model.pkl'))
    vectorizer = joblib.load(os.path.join(models_path, 'tfidf_vectorizer.pkl'))
    label_encoder = joblib.load(os.path.join(models_path, 'label_encoder.pkl'))
    
    with open(os.path.join(models_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    return model, vectorizer, label_encoder, metadata

try:
    model, vectorizer, label_encoder, metadata = load_models()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Prediction function
def predict_category(text):
    cleaned_text = clean_text(text)
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction_idx = model.predict(text_vectorized)[0]
    prediction = label_encoder.inverse_transform([prediction_idx])[0]
    probabilities = model.decision_function(text_vectorized)[0]
    
    prob_dict = {}
    for i, category in enumerate(label_encoder.classes_):
        prob_dict[category] = float(probabilities[i])
    
    total = sum(np.exp(list(prob_dict.values())))
    prob_dict = {k: np.exp(v)/total for k, v in prob_dict.items()}
    
    return prediction, prob_dict

# Advanced analysis functions
def extract_key_terms(text, n=10):
    cleaned = clean_text(text)
    words = cleaned.split()
    word_freq = Counter(words)
    return word_freq.most_common(n)

def calculate_readability_score(text):
    sentences = text.split('.')
    words = text.split()
    
    if len(sentences) == 0 or len(words) == 0:
        return {"words": 0, "sentences": 0, "avg_words_per_sentence": 0, "level": "N/A"}
    
    avg_words = len(words) / len(sentences)
    
    if avg_words < 15:
        level = "Simple"
    elif avg_words < 25:
        level = "Moderate"
    else:
        level = "Complex"
    
    return {
        "words": len(words),
        "sentences": len(sentences),
        "avg_words_per_sentence": round(avg_words, 1),
        "level": level
    }

def analyze_article_structure(text):
    paragraphs = [p for p in text.split('\n') if p.strip()]
    avg_para_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
    
    return {
        "paragraphs": len(paragraphs),
        "avg_paragraph_length": round(avg_para_length, 1)
    }

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">News Article Classifier</h1>
    <p class="main-subtitle">Advanced ML-Powered News Classification System</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="info-box">
        <h3 class="info-box-title">Model Performance</h3>
        <p><strong>Accuracy:</strong> 90.70%</p>
        <p><strong>Precision:</strong> 90.76%</p>
        <p><strong>Recall:</strong> 90.70%</p>
        <p><strong>F1-Score:</strong> 90.68%</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Categories")
    st.markdown("""
    <div>
        <span class="category-badge badge-business">Business</span>
        <span class="category-badge badge-sports">Sports</span>
        <span class="category-badge badge-tech">Tech</span>
        <span class="category-badge badge-world">World</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Dataset Statistics")
    st.markdown("""
    <div class="stats-grid">
        <div class="stat-item">
            <div class="stat-label">Articles</div>
            <div class="stat-value">144.5K</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Features</div>
            <div class="stat-value">10K</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Training</div>
            <div class="stat-value">115.6K</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Testing</div>
            <div class="stat-value">28.9K</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("**IT Business School - ITBS**")
    st.markdown("Machine Learning Project 2024")

# Main Content Tabs
tab1, tab2, tab3 = st.tabs(["📝 Classify Article", "📰 Sample Articles", "📊 Model Details"])

with tab1:
    st.markdown("### Enter Your News Article")
    text_input = st.text_area(
        "",
        placeholder="Paste your news article text here (minimum 10 words)...",
        height=200,
        key="article_input"
    )
    
    if text_input:
        word_count = len(text_input.split())
        char_count = len(text_input)
        st.caption(f"📊 {word_count} words | {char_count} characters")
    
    if st.button("🔍 Classify Article", key="classify_btn"):
        if text_input and len(text_input.split()) >= 10:
            with st.spinner("🤖 Analyzing article..."):
                prediction, probabilities = predict_category(text_input)
                
                st.markdown(f"""
                <div class="prediction-result">
                    <div class="prediction-category">{prediction}</div>
                    <div class="prediction-confidence">
                        Confidence: {probabilities[prediction]*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("#### Confidence Distribution")
                df_prob = pd.DataFrame({
                    'Category': list(probabilities.keys()),
                    'Confidence': [v*100 for v in probabilities.values()]
                })
                df_prob = df_prob.sort_values('Confidence', ascending=True)
                
                fig = px.bar(df_prob, x='Confidence', y='Category', orientation='h',
                            color='Confidence', color_continuous_scale='Blues')
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### Advanced Article Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    readability = calculate_readability_score(text_input)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Readability</div>
                        <div class="metric-value">{readability['level']}</div>
                        <p style="font-size: 0.9rem; color: #64748b; margin-top: 0.5rem;">
                            {readability['avg_words_per_sentence']} words/sentence
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    structure = analyze_article_structure(text_input)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Structure</div>
                        <div class="metric-value">{structure['paragraphs']}</div>
                        <p style="font-size: 0.9rem; color: #64748b; margin-top: 0.5rem;">
                            paragraphs
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                key_terms = extract_key_terms(text_input, n=8)
                if key_terms:
                    st.markdown("#### Key Terms")
                    terms_df = pd.DataFrame(key_terms, columns=['Term', 'Frequency'])
                    fig_terms = px.bar(terms_df, x='Frequency', y='Term', orientation='h')
                    fig_terms.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig_terms, use_container_width=True)
        else:
            st.warning("⚠️ Please enter at least 10 words for accurate classification.")

with tab2:
    st.markdown("### Try These Sample Articles")
    
    samples = {
        "Business": {
            "text": """Wall Street experienced significant volatility today as investors reacted to the Federal Reserve's latest interest rate decision. The S&P 500 index fluctuated throughout the trading session, ultimately closing up 1.2%. Technology stocks led the gains, with major companies reporting better-than-expected quarterly earnings. Analysts suggest that despite concerns about inflation, the overall market sentiment remains cautiously optimistic as corporate profits continue to exceed forecasts.""",
            "class": "business"
        },
        "Sports": {
            "text": """In a thrilling championship match, the defending champions secured their third consecutive title with a dramatic victory in overtime. The star player scored the winning goal in the final minutes, sending fans into celebration. This victory marks a historic achievement for the team, cementing their legacy as one of the greatest dynasties in the sport's history. The coach praised the team's resilience and determination throughout the challenging season.""",
            "class": "sports"
        },
        "Technology": {
            "text": """A breakthrough in artificial intelligence research has scientists excited about the possibilities for quantum computing. Researchers at a leading tech institute have developed a new algorithm that significantly improves machine learning efficiency. The innovation could revolutionize data processing capabilities and accelerate the development of next-generation technologies. Industry experts predict this advancement will have far-reaching implications across multiple sectors, from healthcare to autonomous vehicles.""",
            "class": "tech"
        },
        "World": {
            "text": """International leaders convened at the summit to address pressing global challenges including climate change and economic cooperation. The two-day conference brought together representatives from over 50 nations to negotiate new agreements on environmental protection and sustainable development. Diplomats expressed cautious optimism about reaching consensus on key issues, though significant differences remain on implementation timelines and financial commitments.""",
            "class": "world"
        }
    }
    
    for category, sample in samples.items():
        st.markdown(f"""
        <div class="sample-card {sample['class']}">
            <h4>{category}</h4>
            <p>{sample['text']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"Classify {category} Article", key=f"sample_{category}"):
            prediction, probabilities = predict_category(sample['text'])
            st.success(f"**Predicted Category:** {prediction} ({probabilities[prediction]*100:.1f}% confidence)")

with tab3:
    st.markdown("### Model Configuration")
    
    config_data = {
        "Parameter": ["Algorithm", "C Parameter", "Loss Function", "Max Iterations", "Features", "N-grams"],
        "Value": ["LinearSVC", "0.1", "Squared Hinge", "1000", "10,000", "1-2 (unigrams & bigrams)"]
    }
    st.table(pd.DataFrame(config_data))
    
    st.markdown("### Performance Metrics")
    
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [90.70, 90.76, 90.70, 90.68]
    })
    
    fig_metrics = px.bar(metrics_df, x='Metric', y='Score', color='Metric',
                        color_discrete_sequence=['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981'])
    fig_metrics.update_layout(showlegend=False, yaxis_range=[85, 95])
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    st.markdown("### Per-Class F1-Scores")
    
    f1_scores = pd.DataFrame({
        'Category': ['Business', 'Sports', 'Technology', 'World'],
        'F1-Score': [0.8952, 0.9770, 0.8878, 0.8673]
    })
    
    fig_f1 = px.bar(f1_scores, x='Category', y='F1-Score',
                    color='Category',
                    color_discrete_map={
                        'Business': '#3b82f6',
                        'Sports': '#22c55e',
                        'Technology': '#a855f7',
                        'World': '#eab308'
                    })
    fig_f1.update_layout(showlegend=False, yaxis_range=[0.8, 1.0])
    st.plotly_chart(fig_f1, use_container_width=True)
    
    st.markdown("### Processing Pipeline")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #3b82f6;">1. Text Preprocessing</h4>
            <p style="font-size: 0.9rem; color: #64748b;">
                Lowercase conversion, URL removal, HTML tag cleaning, special character handling
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #8b5cf6;">2. Feature Extraction</h4>
            <p style="font-size: 0.9rem; color: #64748b;">
                TF-IDF vectorization with 10,000 features, 1-2 n-grams, stopword removal
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #06b6d4;">3. Classification</h4>
            <p style="font-size: 0.9rem; color: #64748b;">
                LinearSVC with C=0.1, squared hinge loss, confidence score calculation
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p><strong>IT Business School - ITBS</strong></p>
    <p>Powered by LinearSVC | TF-IDF | scikit-learn</p>
</div>
""", unsafe_allow_html=True)
