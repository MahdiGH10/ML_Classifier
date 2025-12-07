# 📰 News Article Classifier - ML Mini-Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_DEPLOYED_URL_HERE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Automated news article classification system using Machine Learning**  
> Institut Supérieur des Technologies de Bizerte (ITBS) - Machine Learning Mini-Project 2025

![News Classifier Demo](https://img.shields.io/badge/Demo-Live-success)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Methodology](#methodology)
- [Results](#results)
- [Deployment](#deployment)
- [Team](#team)
- [License](#license)

---

## 🎯 Overview

This project implements an **automated news article classification system** that categorizes news articles into 4 categories:

- 📊 **Business** - Financial news, market updates, corporate announcements
- ⚽ **Sports** - Athletic events, team news, player updates  
- 💻 **Technology** - Tech innovations, software, hardware, startups
- 🌍 **World** - International affairs, politics, global events

The system uses **machine learning** with TF-IDF feature extraction and **Support Vector Machines (SVM)** to achieve **90.70% accuracy**.

---

## ✨ Features

✅ **High Accuracy**: 90.70% classification accuracy  
✅ **Fast Predictions**: Real-time article classification  
✅ **User-Friendly Interface**: Clean Streamlit web app  
✅ **Confidence Scores**: Probability distribution for all categories  
✅ **Sample Articles**: Pre-loaded examples for testing  
✅ **Text Statistics**: Word count, character analysis  
✅ **Interactive Visualizations**: Plotly charts  
✅ **Mobile Responsive**: Works on all devices

---

## 🚀 Demo

### Live Application
👉 **[Try the live demo here](YOUR_DEPLOYED_URL_HERE)** 👈

### QR Code
![QR Code](qr_code.png)  
*Scan to access the web app*

### Screenshots

**Main Interface:**
```
┌──────────────────────────────────────────┐
│     📰 News Article Classifier           │
│  Automated Text Classification Using ML  │
├──────────────────────────────────────────┤
│                                          │
│  [Enter article text here...]            │
│                                          │
│          🚀 Classify Article             │
│                                          │
│  Prediction: 💻 Technology               │
│  Confidence: 94.2%                       │
└──────────────────────────────────────────┘
```

---

## 📊 Dataset

### Sources
- **AG News**: 127,600 articles (88.3%)
- **20 Newsgroups**: 16,922 articles (11.7%)
- **Total**: 144,522 articles

### Class Distribution (Balanced)
| Category   | Count   | Percentage |
|------------|---------|------------|
| Business   | 32,821  | 22.7%      |
| Sports     | 35,603  | 24.6%      |
| Technology | 39,381  | 27.2%      |
| World      | 36,717  | 25.4%      |

### Data Split
- **Training**: 70% (101,165 articles)
- **Validation**: 15% (21,678 articles)
- **Test**: 15% (21,679 articles)

---

## 🎯 Model Performance

### Overall Metrics
| Metric              | Score   |
|---------------------|---------|
| **Accuracy**        | 90.70%  |
| **Precision**       | 0.9081  |
| **Recall**          | 0.9083  |
| **F1-Score**        | 0.9068  |

### Per-Class Performance
| Category   | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Business   | 0.89      | 0.87   | 0.88     | 4,923   |
| Sports     | 0.95      | 0.97   | 0.96     | 5,340   |
| Technology | 0.89      | 0.90   | 0.89     | 5,908   |
| World      | 0.91      | 0.90   | 0.91     | 5,508   |

**Key Insights:**
- ✅ Sports articles are easiest to classify (F1=0.96)
- ⚠️ Business-Technology confusion due to tech company news
- ✅ All categories exceed F1-score of 0.87

---

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/news-classifier.git
cd news-classifier
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

5. **Run the application**
```bash
streamlit run app.py
```

6. **Open browser**
```
Local URL: http://localhost:8501
Network URL: http://YOUR_IP:8501
```

---

## 📖 Usage

### Web Interface

1. **Launch the app**: `streamlit run app.py`
2. **Enter text**: Paste a news article in the text area
3. **Click "Classify Article"**: Get instant prediction
4. **View results**: See category, confidence scores, and statistics

### Programmatic Usage

```python
import joblib
import re
from nltk.corpus import stopwords

# Load models
model = joblib.load('FinalModel/models_4class/best_model.pkl')
vectorizer = joblib.load('FinalModel/models_4class/tfidf_vectorizer.pkl')
label_encoder = joblib.load('FinalModel/models_4class/label_encoder.pkl')

# Preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

# Predict
article = "Your news article text here..."
cleaned = clean_text(article)
features = vectorizer.transform([cleaned])
prediction = model.predict(features)[0]
category = label_encoder.inverse_transform([prediction])[0]

print(f"Category: {category}")
```

---

## 📁 Project Structure

```
ML_Classifier/
├── src/
│   └── app_enhanced.py          # Main Streamlit application
├── models/
│   └── FinalModel/
│       └── models_4class/
│           ├── best_model.pkl          # Trained LinearSVC model
│           ├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
│           ├── label_encoder.pkl       # Label encoder
│           ├── categories.txt          # Category names
│           └── metadata.json           # Model metadata
├── notebooks/
│   └── Ml_Classiferfinal.ipynb  # Training notebook
├── data/
│   └── README.md                 # Dataset information
├── assets/
│   └── background.png            # UI background image
├── Rapport/                      # LaTeX report
│   ├── rapport.tex              # Main report file
│   ├── introduction.tex         # Introduction
│   ├── chapitre1.tex           # Dataset & Preprocessing
│   ├── chapitre2.tex           # Methodology & Results
│   ├── abstract.tex            # Abstract (EN/FR)
│   ├── Biblio.bib              # Bibliography
│   └── poster.tex              # Scientific poster
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
├── DEPLOYMENT_GUIDE.md         # Deployment instructions
├── generate_qr.py              # QR code generator
└── README.md                    # This file
```

---

## 🛠️ Technologies Used

### Machine Learning
- **Scikit-learn** 1.3.2 - ML algorithms, TF-IDF, metrics
- **NLTK** 3.8.1 - Text preprocessing, stopwords
- **NumPy** 1.24.3 - Numerical computations
- **Pandas** 2.0.3 - Data manipulation

### Web Application
- **Streamlit** 1.31.0 - Interactive web interface
- **Plotly** 5.18.0 - Interactive visualizations

### Deployment
- **Streamlit Cloud** - Cloud hosting
- **Git/GitHub** - Version control
- **Joblib** 1.3.2 - Model serialization

---

## 🔬 Methodology

### 1. Data Preprocessing
```
Raw Text → Cleaning → Normalization → Stopword Removal → Cleaned Text
```

**Steps:**
- Convert to lowercase
- Remove URLs, HTML tags, special characters
- Remove stopwords (NLTK English stopwords)
- Remove texts shorter than 20 characters

### 2. Feature Extraction (TF-IDF)
```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),      # Unigrams + Bigrams
    min_df=5,                # Min document frequency
    max_df=0.8,              # Max document frequency
    sublinear_tf=True
)
```

**Formula:**
```
TF-IDF(t,d) = TF(t,d) × IDF(t)
where:
  TF(t,d) = frequency of term t in document d
  IDF(t) = log(N / df(t))
  N = total documents
  df(t) = documents containing term t
```

### 3. Model Training

**Algorithms Tested:**
1. Logistic Regression (C=0.3)
2. **LinearSVC** (C=0.1, squared_hinge) ← **Winner**
3. Random Forest (200 estimators)

**Hyperparameter Tuning:**
- 5-fold stratified cross-validation
- Grid search optimization
- Scoring metric: F1-weighted

### 4. Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curves (one-vs-rest)
- Feature Importance Analysis

---

## 📈 Results

### Confusion Matrix (LinearSVC)
```
                Predicted
            Bus   Spo   Tech  Wld
Actual Bus  4263  23    402   235
       Spo  26    5155  50    109
       Tech 296   74    5318  220
       Wld  245   98    221   4944
```

### Key Findings
✅ **Sports** easiest to classify (distinct vocabulary)  
⚠️ **Business-Tech** confusion (tech companies, startups)  
✅ **Balanced** performance across all classes  
✅ **Target exceeded**: F1 = 0.907 > 0.85

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🏫 Academic Project

**Institution**: IT Business School (ITBS)  
**Course**: Machine Learning  
**Year**: 2024-2025  
**Type**: Mini-Project

---

## 👥 Authors

- **Mahdi GH** - [GitHub](https://github.com/MahdiGH10)

---

## 🙏 Acknowledgments

- AG News Dataset
- 20 Newsgroups Dataset
- Streamlit for the amazing framework
- scikit-learn for ML tools
- ITBS faculty for guidance

---

## 📧 Contact

For questions or feedback:
- GitHub: [@MahdiGH10](https://github.com/MahdiGH10)
- Repository: [ML_Classifier](https://github.com/MahdiGH10/ML_Classifier)

---

## 🔗 Links

- [Live Demo](https://your-deployment-url.streamlit.app) *(coming soon)*
- [Documentation](https://github.com/MahdiGH10/ML_Classifier/wiki)
- [Report Issues](https://github.com/MahdiGH10/ML_Classifier/issues)

---

⭐ **Star this repository if you find it helpful!**

Made with ❤️ by ITBS Students
