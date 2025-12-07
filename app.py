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

# Custom CSS - Professional Design
st.markdown("""
<style>
    /* Import Professional Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Container */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header Styles */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Card Styles */
    .card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Prediction Result */
    .prediction-result {
        background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .prediction-category {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .prediction-confidence {
        font-size: 1.2rem;
        font-weight: 500;
        opacity: 0.95;
    }
    
    /* Metric Cards */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
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
        color: #1e293b;
    }
    
    /* Button Styles */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.875rem 2rem;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.35);
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    
    /* Sidebar Styles */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Info Box */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
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
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
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
    
    /* Text Input */
    .stTextArea textarea {
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        font-size: 1rem;
        padding: 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
    }
    
    /* Progress Indicator */
    .analysis-progress {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #64748b;
        font-size: 0.875rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 3rem;
    }
    
    /* Category Badges with hover effects */
    .category-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.25rem;
        transition: all 0.3s ease;
        cursor: default;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .category-badge:hover {
        transform: translateY(-2px) scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
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
</style>
""", unsafe_allow_html=True)

# Load models and metadata
@st.cache_resource
def load_models():
    """Load trained models and metadata"""
    try:
        model = joblib.load('FinalModel/models_4class/best_model.pkl')
        vectorizer = joblib.load('FinalModel/models_4class/tfidf_vectorizer.pkl')
        label_encoder = joblib.load('FinalModel/models_4class/label_encoder.pkl')
        
        with open('FinalModel/models_4class/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return model, vectorizer, label_encoder, metadata
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

model, vectorizer, label_encoder, metadata = load_models()

# Text preprocessing
def clean_text(text):
    """Clean and preprocess text"""
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        words = text.split()
        text = ' '.join([w for w in words if w not in stop_words])
    except:
        pass
    
    return text

def predict_category(text):
    """Predict news category"""
    if not text.strip():
        return None, None
    
    # Clean text
    cleaned = clean_text(text)
    
    # Transform with TF-IDF
    features = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(features)[0]
    
    # Get probabilities (if available)
    if hasattr(model, 'decision_function'):
        decision_scores = model.decision_function(features)[0]
        # Convert to pseudo-probabilities using softmax
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        probabilities = exp_scores / exp_scores.sum()
    else:
        probabilities = model.predict_proba(features)[0]
    
    # Get category name
    category = label_encoder.inverse_transform([prediction])[0]
    
    return category, probabilities

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">News Article Classifier</div>
    <div class="main-subtitle">Advanced Text Classification Using Machine Learning</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Model Information")
    
    if metadata:
        st.markdown(f"""
        <div class="info-box">
            <div class="info-box-title">Model Performance</div>
            <div style="margin-top: 0.75rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #64748b;">Algorithm</span>
                    <span style="font-weight: 600;">{metadata['model_name']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #64748b;">Accuracy</span>
                    <span style="font-weight: 600; color: #059669;">{metadata['test_accuracy']*100:.2f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #64748b;">F1-Score</span>
                    <span style="font-weight: 600; color: #059669;">{metadata['test_f1_weighted']:.4f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Classification Categories")
    st.markdown("""
    <div style="margin-top: 1rem;">
        <span class="category-badge badge-business">Business</span>
        <span class="category-badge badge-sports">Sports</span>
        <span class="category-badge badge-tech">Technology</span>
        <span class="category-badge badge-world">World</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Dataset")
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-item">
            <div class="stat-label">Total Articles</div>
            <div class="stat-value">144.5K</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Features</div>
            <div class="stat-value">10K</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Training Set</div>
            <div class="stat-value">{metadata['train_size']:,}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Test Set</div>
            <div class="stat-value">{metadata['test_size']:,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">IT Business School</div>
        <div style="color: #64748b; font-size: 0.875rem;">ITBS</div>
        <div style="color: #64748b; font-size: 0.875rem; margin-top: 0.25rem;">Machine Learning Project 2025</div>
    </div>
    """, unsafe_allow_html=True)

# Advanced Analysis Functions
def extract_key_terms(text, n=10):
    """Extract most frequent meaningful words"""
    cleaned = clean_text(text)
    words = cleaned.split()
    # Filter out very short words
    words = [w for w in words if len(w) > 3]
    word_freq = Counter(words)
    return word_freq.most_common(n)

def calculate_readability_score(text):
    """Simple readability metrics"""
    sentences = text.count('.') + text.count('!') + text.count('?')
    words = len(text.split())
    chars = len(text)
    
    if sentences == 0 or words == 0:
        return 0, 0, 0
    
    avg_words_per_sentence = words / sentences if sentences > 0 else 0
    avg_chars_per_word = chars / words if words > 0 else 0
    
    return words, sentences, avg_words_per_sentence

def analyze_article_structure(text):
    """Analyze article structure"""
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    num_paragraphs = len(paragraphs)
    
    # Calculate avg paragraph length
    if num_paragraphs > 0:
        avg_para_length = sum(len(p.split()) for p in paragraphs) / num_paragraphs
    else:
        avg_para_length = 0
    
    return num_paragraphs, avg_para_length

# Main content
tab1, tab2, tab3 = st.tabs(["Classify Article", "Sample Articles", "Model Details"])

with tab1:
    st.markdown('<div class="card"><div class="card-title">Article Classification</div>', unsafe_allow_html=True)
    
    # Input section
    st.markdown("##### Enter or paste your news article below")
    st.markdown('<p style="color: #64748b; font-size: 0.875rem; margin-bottom: 1rem;">For optimal results, provide at least 50 words of article text.</p>', unsafe_allow_html=True)
    
    text_input = st.text_area(
        "",
        height=250,
        placeholder="Paste your news article text here...\n\nExample: Technology companies are investing heavily in artificial intelligence research...",
        label_visibility="collapsed"
    )
    
    # Character and word count
    if text_input:
        word_count = len(text_input.split())
        char_count = len(text_input)
        st.markdown(f'<p style="color: #64748b; font-size: 0.875rem; text-align: right;">{word_count} words | {char_count} characters</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Classify button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("Analyze Article", type="primary")
    
    if predict_button and text_input:
        if len(text_input.split()) < 10:
            st.error("⚠ Please enter at least 10 words for accurate classification.")
        else:
            with st.spinner("Processing article..."):
                category, probabilities = predict_category(text_input)
                
                if category:
                    # Category color mapping
                    category_colors = {
                        "Business": ("#dbeafe", "#1e40af"),
                        "Sports": ("#dcfce7", "#166534"),
                        "Tech": ("#f3e8ff", "#6b21a8"),
                        "World": ("#fef3c7", "#92400e")
                    }
                    
                    bg_color, text_color = category_colors.get(category, ("#f1f5f9", "#1e293b"))
                    confidence = probabilities[list(metadata['categories']).index(category)] * 100
                    
                    # Display prediction result
                    st.markdown(f"""
                    <div class="prediction-result">
                        <div style="font-size: 0.875rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9; margin-bottom: 0.5rem;">Predicted Category</div>
                        <div class="prediction-category">{category}</div>
                        <div class="prediction-confidence">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                    # Confidence Distribution
                    st.markdown('<div class="card"><div class="card-title">Confidence Distribution</div>', unsafe_allow_html=True)
                    
                    categories = metadata['categories']
                    conf_df = pd.DataFrame({
                        'Category': categories,
                        'Confidence': probabilities * 100
                    }).sort_values('Confidence', ascending=False)
                    
                    # Create horizontal bar chart
                    fig = go.Figure()
                    
                    colors = ['#3b82f6' if cat == category else '#e2e8f0' for cat in conf_df['Category']]
                    
                    fig.add_trace(go.Bar(
                        x=conf_df['Confidence'],
                        y=conf_df['Category'],
                        orientation='h',
                        marker=dict(color=colors),
                        text=[f'{x:.1f}%' for x in conf_df['Confidence']],
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        showlegend=False,
                        height=280,
                        margin=dict(l=0, r=0, t=20, b=0),
                        xaxis=dict(
                            title="Confidence Score (%)",
                            range=[0, 105],
                            gridcolor='#f1f5f9'
                        ),
                        yaxis=dict(title=""),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family="Inter, sans-serif", size=12)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Advanced Article Analysis
                    st.markdown('<div class="card"><div class="card-title">Article Analysis</div>', unsafe_allow_html=True)
                    
                    # Calculate metrics
                    words, sentences, avg_words = calculate_readability_score(text_input)
                    num_paragraphs, avg_para_length = analyze_article_structure(text_input)
                    key_terms = extract_key_terms(text_input, n=10)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Text Statistics")
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-card">
                                <div class="metric-label">Word Count</div>
                                <div class="metric-value">{words}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Sentences</div>
                                <div class="metric-value">{sentences}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Paragraphs</div>
                                <div class="metric-value">{num_paragraphs}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Avg Words/Sentence</div>
                                <div class="metric-value">{avg_words:.1f}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Readability assessment
                        if avg_words < 15:
                            readability = "Simple"
                            read_color = "#059669"
                        elif avg_words < 25:
                            readability = "Moderate"
                            read_color = "#0284c7"
                        else:
                            readability = "Complex"
                            read_color = "#dc2626"
                        
                        st.markdown(f"""
                        <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 4px solid {read_color};">
                            <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.25rem;">Readability</div>
                            <div style="color: {read_color}; font-weight: 600; font-size: 1.1rem;">{readability}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("##### Key Terms")
                        if key_terms:
                            # Create term frequency chart
                            terms_df = pd.DataFrame(key_terms, columns=['Term', 'Frequency'])
                            
                            fig_terms = go.Figure()
                            fig_terms.add_trace(go.Bar(
                                x=terms_df['Frequency'],
                                y=terms_df['Term'],
                                orientation='h',
                                marker=dict(
                                    color=terms_df['Frequency'],
                                    colorscale='Blues',
                                    showscale=False
                                ),
                                text=terms_df['Frequency'],
                                textposition='outside',
                                hovertemplate='<b>%{y}</b><br>Frequency: %{x}<extra></extra>'
                            ))
                            
                            fig_terms.update_layout(
                                height=300,
                                margin=dict(l=0, r=0, t=10, b=0),
                                xaxis=dict(title="Frequency", gridcolor='#f1f5f9'),
                                yaxis=dict(title=""),
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                font=dict(family="Inter, sans-serif", size=11)
                            )
                            
                            st.plotly_chart(fig_terms, use_container_width=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    elif predict_button:
        st.warning("⚠️ Please enter some text to classify!")

with tab2:
    st.markdown('<div class="card"><div class="card-title">Sample News Articles</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; margin-bottom: 2rem;">Try these example articles to see the classifier in action.</p>', unsafe_allow_html=True)
    
    examples = {
        "Business": {
            "title": "Tech Giants Report Record Quarterly Earnings",
            "text": "Major technology companies reported unprecedented quarterly earnings, driven by strong demand for cloud computing services and artificial intelligence solutions. Apple, Microsoft, and Alphabet exceeded Wall Street expectations, with combined revenues surpassing $300 billion. Market analysts attribute the growth to enterprise digital transformation initiatives and consumer spending on premium devices. The tech sector's robust performance has sparked renewed investor confidence despite global economic uncertainties.",
            "icon": "business"
        },
        "Sports": {
            "title": "Olympic Champions Crowned in Dramatic Final",
            "text": "The national team clinched the gold medal in a thrilling Olympic final that went down to the wire. Star athlete Sarah Johnson delivered a record-breaking performance in the final seconds, securing victory against fierce international competition. The triumph marks the country's third consecutive Olympic title in this event. Fans celebrated in the streets as the team's dedication and months of rigorous training paid off on the world's biggest sporting stage.",
            "icon": "sports"
        },
        "Tech": {
            "title": "Breakthrough in Quantum Computing Announced",
            "text": "Researchers unveiled a revolutionary quantum computing processor capable of solving complex problems exponentially faster than traditional supercomputers. The breakthrough, published in Nature, demonstrates sustained quantum coherence for over 100 qubits. Scientists predict this advancement will accelerate drug discovery, optimize financial modeling, and revolutionize cryptography. Major tech companies are already partnering with research institutions to commercialize the technology within the next five years.",
            "icon": "tech"
        },
        "World": {
            "title": "International Climate Agreement Reached at Summit",
            "text": "World leaders concluded a landmark climate summit with a comprehensive agreement to reduce global carbon emissions by 50% within the next decade. The accord includes binding commitments from 190 nations, unprecedented funding for renewable energy infrastructure, and support for developing countries' transition to clean energy. Environmental organizations praised the deal as a crucial step toward limiting global temperature rise, though some activists called for more aggressive targets.",
            "icon": "world"
        }
    }
    
    for idx, (cat, content) in enumerate(examples.items()):
        category_colors = {
            "Business": "#dbeafe",
            "Sports": "#dcfce7",
            "Tech": "#f3e8ff",
            "World": "#fef3c7"
        }
        
        st.markdown(f"""
        <div style="background: {category_colors.get(cat, '#f8fafc')}; padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; border: 1px solid #e2e8f0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div>
                    <div style="font-size: 0.875rem; font-weight: 600; text-transform: uppercase; color: #64748b; letter-spacing: 0.5px;">{cat}</div>
                    <div style="font-size: 1.25rem; font-weight: 600; color: #1e293b; margin-top: 0.25rem;">{content['title']}</div>
                </div>
            </div>
            <div style="color: #475569; line-height: 1.6; margin-bottom: 1rem;">{content['text']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"Classify this {cat} article", key=f"classify_{idx}"):
            with st.spinner("Analyzing..."):
                category, probabilities = predict_category(content['text'])
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    confidence = probabilities[list(metadata['categories']).index(category)] * 100
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%); color: white; padding: 1.5rem; border-radius: 8px; text-align: center;">
                        <div style="font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem;">Predicted Category</div>
                        <div style="font-size: 1.75rem; font-weight: 700;">{category}</div>
                        <div style="font-size: 0.95rem; margin-top: 0.5rem;">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card"><div class="card-title">Model Architecture & Performance</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Model Configuration")
        st.markdown(f"""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; border: 1px solid #e2e8f0;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 0.75rem 0; color: #64748b; font-weight: 500;">Algorithm</td>
                    <td style="padding: 0.75rem 0; color: #1e293b; font-weight: 600; text-align: right;">{metadata['model_name']}</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 0.75rem 0; color: #64748b; font-weight: 500;">Regularization (C)</td>
                    <td style="padding: 0.75rem 0; color: #1e293b; font-weight: 600; text-align: right;">{metadata['best_params']['C']}</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 0.75rem 0; color: #64748b; font-weight: 500;">Loss Function</td>
                    <td style="padding: 0.75rem 0; color: #1e293b; font-weight: 600; text-align: right;">{metadata['best_params']['loss']}</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 0.75rem 0; color: #64748b; font-weight: 500;">Max Iterations</td>
                    <td style="padding: 0.75rem 0; color: #1e293b; font-weight: 600; text-align: right;">{metadata['best_params']['max_iter']:,}</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 0.75rem 0; color: #64748b; font-weight: 500;">TF-IDF Features</td>
                    <td style="padding: 0.75rem 0; color: #1e293b; font-weight: 600; text-align: right;">{metadata['num_features']:,}</td>
                </tr>
                <tr>
                    <td style="padding: 0.75rem 0; color: #64748b; font-weight: 500;">Hyperparameter Tuning</td>
                    <td style="padding: 0.75rem 0; color: #1e293b; font-weight: 600; text-align: right;">5-Fold CV</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("##### Training Data")
        st.markdown(f"""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-top: 1rem;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 0.75rem 0; color: #64748b; font-weight: 500;">Training Set</td>
                    <td style="padding: 0.75rem 0; color: #1e293b; font-weight: 600; text-align: right;">{metadata['train_size']:,} articles</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0;">
                    <td style="padding: 0.75rem 0; color: #64748b; font-weight: 500;">Test Set</td>
                    <td style="padding: 0.75rem 0; color: #1e293b; font-weight: 600; text-align: right;">{metadata['test_size']:,} articles</td>
                </tr>
                <tr>
                    <td style="padding: 0.75rem 0; color: #64748b; font-weight: 500;">Total Dataset</td>
                    <td style="padding: 0.75rem 0; color: #1e293b; font-weight: 600; text-align: right;">144,522 articles</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("##### Performance Metrics")
        
        # Performance metrics
        perf_data = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Score': [90.70, 90.81, 90.83, 90.68]
        })
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Bar(
            x=perf_data['Score'],
            y=perf_data['Metric'],
            orientation='h',
            marker=dict(
                color=['#3b82f6', '#0ea5e9', '#06b6d4', '#10b981'],
                line=dict(color='#1e293b', width=1)
            ),
            text=[f'{x:.2f}%' for x in perf_data['Score']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Score: %{x:.2f}%<extra></extra>'
        ))
        
        fig_perf.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(range=[0, 100], title="Score (%)", gridcolor='#f1f5f9'),
            yaxis=dict(title=""),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12)
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
        
        st.markdown("##### Per-Class F1-Scores")
        
        class_data = pd.DataFrame({
            'Category': ['Sports', 'World', 'Technology', 'Business'],
            'F1-Score': [96, 91, 89, 88]
        })
        
        fig_class = go.Figure()
        fig_class.add_trace(go.Bar(
            x=class_data['F1-Score'],
            y=class_data['Category'],
            orientation='h',
            marker=dict(color='#3b82f6'),
            text=[f'{x}%' for x in class_data['F1-Score']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>F1-Score: %{x}%<extra></extra>'
        ))
        
        fig_class.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(range=[0, 105], title="F1-Score (%)", gridcolor='#f1f5f9'),
            yaxis=dict(title=""),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=12)
        )
        
        st.plotly_chart(fig_class, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Pipeline Section
    st.markdown('<div class="card" style="margin-top: 1.5rem;"><div class="card-title">Processing Pipeline</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); padding: 1.5rem; border-radius: 10px; height: 100%;">
            <div style="font-weight: 600; color: #1e40af; font-size: 1.1rem; margin-bottom: 1rem;">1. Text Preprocessing</div>
            <ul style="color: #1e40af; font-size: 0.9rem; line-height: 1.8; padding-left: 1.2rem;">
                <li>Lowercase conversion</li>
                <li>URL & HTML removal</li>
                <li>Special character filtering</li>
                <li>Stopword elimination</li>
                <li>Whitespace normalization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 1.5rem; border-radius: 10px; height: 100%;">
            <div style="font-weight: 600; color: #0369a1; font-size: 1.1rem; margin-bottom: 1rem;">2. Feature Extraction</div>
            <ul style="color: #0369a1; font-size: 0.9rem; line-height: 1.8; padding-left: 1.2rem;">
                <li>TF-IDF vectorization</li>
                <li>10,000 features max</li>
                <li>Unigrams + Bigrams</li>
                <li>Min DF: 5 documents</li>
                <li>Max DF: 80% documents</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); padding: 1.5rem; border-radius: 10px; height: 100%;">
            <div style="font-weight: 600; color: #065f46; font-size: 1.1rem; margin-bottom: 1rem;">3. Classification</div>
            <ul style="color: #065f46; font-size: 0.9rem; line-height: 1.8; padding-left: 1.2rem;">
                <li>LinearSVC algorithm</li>
                <li>Grid search optimization</li>
                <li>5-fold cross-validation</li>
                <li>Stratified sampling</li>
                <li>F1-weighted scoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <div style="font-weight: 600; color: #1e293b; font-size: 1.1rem; margin-bottom: 0.5rem;">IT Business School - ITBS</div>
    <div style="color: #64748b; margin-bottom: 0.25rem;">Machine Learning Mini-Project 2025</div>
    <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 1rem;">
        Built with Streamlit | Powered by Scikit-learn | Model: LinearSVC
    </div>
</div>
""", unsafe_allow_html=True)
