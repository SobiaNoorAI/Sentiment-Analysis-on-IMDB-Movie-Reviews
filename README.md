# 🎬 Sentiment Analysis of IMDb Movie Reviews

This project analyzes IMDb movie reviews to determine whether a review is **positive** or **negative** using **Natural Language Processing (NLP)** techniques and classification models. It includes data preprocessing, feature extraction (BoW & TF-IDF), exploratory data analysis (EDA), model building using **Naïve Bayes**, and evaluation.

---

## 📌 Table of Contents

- [📌 Table of Contents](#-table-of-contents)
- [🎯 Objective](#-objective)
- [📂 Dataset Description](#-dataset-description)
- [🔍 Project Workflow](#-project-workflow)
  - [1️⃣ Data Preprocessing](#1️⃣-data-preprocessing)
  - [2️⃣ Exploratory Data Analysis (EDA)](#2️⃣-exploratory-data-analysis-eda)
  - [3️⃣ Feature Engineering](#3️⃣-feature-engineering)
  - [4️⃣ Model Building](#4️⃣-model-building)
- [📊 Visualizations](#-visualizations)
- [🚀 Tech Stack](#-tech-stack)
- [📈 Results](#-results)
- [📁 Project Structure](#-project-structure)
- [📦 Installation Guide](#-installation-guide)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🎯 Objective

To classify IMDb movie reviews as **positive** or **negative** using NLP and machine learning techniques.

---

## 📂 Dataset Description

- **Dataset:** IMDb Movie Reviews  
- **Total Samples:** 50,000 reviews  
- **Classes:**  
  - `Positive`  
  - `Negative`

The dataset contains an equal number of positive and negative reviews.

---

## 🔍 Project Workflow

### 1️⃣ Data Preprocessing

- Removal of HTML tags
- Lowercasing all text
- Removing punctuation, numbers, and special characters
- Removing stopwords
- Applying stemming using NLTK’s `PorterStemmer`

---

### 2️⃣ Exploratory Data Analysis (EDA)

- Distribution of sentiment labels
- Review length distribution
- Most frequent words in each sentiment
- Word clouds for positive and negative reviews

---

### 3️⃣ Feature Engineering

- **Bag of Words (BoW)** model using `CountVectorizer`
- **TF-IDF** model using `TfidfVectorizer`

---

### 4️⃣ Model Building

- Data split: 80% training, 20% testing
- Model: **Multinomial Naïve Bayes**
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix

---

## 📊 Visualizations

- Sentiment distribution (Bar Plot)
- Word clouds for positive & negative reviews
- Confusion matrix heatmap

---

## 🚀 Tech Stack

| Category            | Libraries / Tools                    |
|---------------------|--------------------------------------|
| Language            | Python                               |
| NLP & Text Prep     | NLTK, re, string                     |
| Feature Extraction  | CountVectorizer, TfidfVectorizer     |
| Model Building      | Scikit-learn (Naïve Bayes)           |
| Visualization       | Matplotlib, Seaborn, WordCloud       |
| IDE/Notebook        | Jupyter Notebook / Colab / VS Code   |

---

## 📈 Results

| Metric     | Value (Example) |
|------------|-----------------|
| Accuracy   | 85%             |
| Precision  | 84%             |
| Recall     | 86%             |

---

## 📁 Project Structure
IMDb-Sentiment-Analysis/
│
├── data/
│ └── IMDB Dataset.csv
├── notebooks/
│ └── Sentiment_Analysis_By_Classification_Model.ipynb
├── LICENSE
└── README.md

---

## 🔹 Requirements
Ensure you have the following Python libraries installed:

pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud
## 📦 Installation Guide

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Sentiment-Analysis-on-IMDB-Movie-Reviews.git
cd Sentiment-Analysis-on-IMDB-Movie-Reviews
## 🤝 Contributing
Feel free to fork the repo, submit issues, or create pull requests.
## 📄 License
Distributed under the MIT License. See LICENSE for more information
