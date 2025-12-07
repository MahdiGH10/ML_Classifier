# 📊 Dataset Information

## Overview

This project uses a combination of two well-known news classification datasets:

### 1. AG News Dataset
- **Size**: 127,600 articles
- **Categories**: Business, Sports, Technology, World
- **Source**: Academic news corpus
- **Format**: Pre-processed text

### 2. 20 Newsgroups Dataset
- **Size**: 16,922 articles (filtered for relevant categories)
- **Categories**: Mapped to Business, Sports, Technology, World
- **Source**: Usenet newsgroups
- **Format**: Raw text

## Combined Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Articles** | 144,522 |
| **Training Set** | 115,618 (80%) |
| **Test Set** | 28,904 (20%) |
| **Average Article Length** | ~150 words |
| **Vocabulary Size** | 10,000 features |

## Category Distribution

```
Business:    32,821 articles (22.7%)
Sports:      35,603 articles (24.6%)
Technology:  39,381 articles (27.2%)
World:       36,717 articles (25.4%)
```

## Data Privacy

⚠️ **Note**: Due to size constraints, the raw dataset files are not included in this repository.

To use this project:
1. The trained model is included (`models/FinalModel/`)
2. For retraining, download datasets from:
   - AG News: [Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)
   - 20 Newsgroups: Included in scikit-learn

## Preprocessing Steps

1. Text lowercasing
2. URL removal
3. HTML tag removal
4. Special character cleaning
5. Stopword removal
6. Whitespace normalization
7. TF-IDF vectorization (10K features, 1-2 n-grams)

## License

- **AG News**: CC BY-SA 3.0
- **20 Newsgroups**: Public Domain
