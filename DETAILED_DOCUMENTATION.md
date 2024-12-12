# Personalized Ad Recommendation System

A personalized ad recommendation system that uses sentiment analysis to recommend products based on user reviews. This project leverages pre-trained sentiment analysis models and collaborative filtering techniques to provide tailored product recommendations to users.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [How to Use the Software](#how-to-use-the-software)  
   - [Installation](#installation)  
   - [Running the Application](#running-the-application)  
   - [Sample User IDs](#sample-user-ids)  
3. [System Architecture](#system-architecture)  
4. [Code Implementation](#code-implementation)  
   - [Data Preprocessing](#data-preprocessing)  
   - [Sentiment Analysis](#sentiment-analysis)  
   - [Recommendation Engine](#recommendation-engine)  
   - [Streamlit App Interface](#streamlit-app-interface)  
5. [Project Structure](#project-structure)  
6. [Troubleshooting](#troubleshooting)
7. [License](#license)  
8. [Contact](#contact)

---

## Introduction

This project builds a **Personalized Ad Recommendation System** that uses sentiment analysis on user reviews to recommend products. The system leverages pre-trained transformer models for sentiment classification and collaborative filtering techniques for generating recommendations.

---

## How to Use the Software

### Installation

1. **Clone the GitHub Repository**:

   ```bash
    git clone https://github.com/jinxuanyao/CS410_Project.git
    cd CS410_Project
   ```

2. **（optional)Set Up a Virtual Environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate       # On macOS/Linux
   .venv\Scripts\activate          # On Windows
   ```

3. **Install Dependencies**:

   ```bash
   pip install torch transformers pandas scikit-learn numpy scipy matplotlib seaborn jupyter streamlit flask
   ```

### Running the Application

1. **Launch the Streamlit App**:

   ```bash
   streamlit run src/app.py
   ```

2. **Interact with the App**:

   - Enter a **User ID** in the input field.
   - Click **"Get Recommendations"** to receive personalized product recommendations.
   - View the sentiment distribution charts for each recommended product.

### Sample User IDs

Here are some sample User IDs you can use for testing:

- **`A1P21J0DMTVGS7`**  
- **`A1ZKFQLHFZAEH9`**  
- **`A2NO1TXXS9T0EE`**  

---

## System Architecture

The system consists of the following main components:

1. **Data Preprocessing Module**:
   - Cleans and splits the dataset into training and testing sets.
   
2. **Sentiment Analysis Module**:
   - Uses a pre-trained transformer model to classify the sentiment of user reviews.

3. **Recommendation Engine**:
   - Implements collaborative filtering to recommend products based on user interactions.

4. **Streamlit Web Interface**:
   - Provides an interactive user interface for inputting User IDs and displaying recommendations.

---

## Code Implementation

### Data Preprocessing

**File**: `src/data_preprocessing.py`

**Functionality**:

1. **Load Data**: Reads the dataset from `data/Reviews.csv`.  
2. **Clean Data**: Removes rows with missing `review_text` or `rating`. Rename Data. 
3. **Split Data**: Divides the data into `train.csv` and `test.csv` (80% train, 20% test).  

**Key Code**:

```python
def load_and_clean_data(input_path, sample_size=100):
    df = pd.read_csv(input_path)

    selected_columns = df[['UserId', 'ProductId', 'Text', 'Score']]

    selected_columns.columns = ['user_id', 'product_id', 'review_text', 'rating']

    selected_columns.dropna(subset=['review_text', 'rating'], inplace=True)

    selected_columns['rating'] = selected_columns['rating'].astype(int)

    sampled_df = selected_columns.sample(n=sample_size, random_state=16)

    return sampled_df

def split_data(df):
    train, test = train_test_split(df, test_size=0.2, random_state=16)
    return train, test
```

---

### Sentiment Analysis

**File**: `src/sentiment_analysis.py`

**Functionality**:

1. **Load Pre-trained Model**: Uses `transformers` library to load a sentiment analysis model.  
2. **Predict Sentiment**: Classifies reviews as `Very Negative`, `Negative`, `Neutral`, `Positive`, or `Very Positive`.  

**Key Code**:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentAnalyzer:
    def __init__(self, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
```

---

### Recommendation Engine

**File**: `src/recommendation.py`

**Functionality**:

1. **Collaborative Filtering**: Recommends products based on user interactions.  

**Key Code**:

```python
class Recommender:
    def __init__(self, data):
        self.data = data

    def recommend(self, user_id):
        user_data = self.data[self.data['user_id'] == user_id]
        return user_data['product_id'].unique().tolist()
```

---

### Streamlit App Interface

**File**: `src/app.py`

**Functionality**:

1. **Input User ID**.  
2. **Display Recommendations**.  
3. **Visualize Sentiment Distribution**.  

**Key Code**:

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from recommendation import Recommender

st.title("Personalized Ad Recommendation System")

train = pd.read_csv("data/train_with_sentiment.csv")
recommender = Recommender(train)

user_id_input = st.text_input("Enter User ID:", "12345")

if st.button("Get Recommendations"):
    recs = recommender.recommend(user_id_input)
    if recs:
        st.write(f"Recommended products for User {user_id_input}:")
        for product_id in recs:
            st.write(f"- {product_id}")
```

---

## Project Structure

```
pythonProject/
│-- data/
│   ├── Reviews.csv
│   ├── train.csv
│   ├── test.csv
│   └── train_with_sentiment.csv
│-- src/
│   ├── app.py
│   ├── data_preprocessing.py
│   ├── recommendation.py
│   └── sentiment_analysis.py
│-- create_sentiment.py
│-- requirements.txt
│-- LICENSE
└-- README.md
```

---

## Troubleshooting

1. **Model Loading Issues**: Ensure `transformers` and `torch` are installed.
2. **Memory Errors**: Try reducing batch size during sentiment analysis.
3. **Streamlit Errors**: Ensure Streamlit is installed and virtual environment is activated.


---

## License

MIT License.

---

## Contact

- **Name**: Joanna Jinxuan Yao  
- **Email**: jinxuan2@illinois.edu  
