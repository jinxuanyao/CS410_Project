# Personalized Ad Recommendation System

A personalized ad recommendation system that uses sentiment analysis to recommend products based on user reviews. This project leverages pre-trained sentiment analysis models and collaborative filtering techniques to provide tailored product recommendations to users.

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Installation](#installation)  
4. [Usage](#usage)
5. [Project Structure](#project-structure)  
6. [License](#license)  

---

## Project Overview

This project aims to build a recommendation system that suggests products to users based on their past interactions and sentiment analysis of product reviews. The system uses the following key steps:

1. **Sentiment Analysis**: Classifies user reviews as "Very Negative", "Negative", "Neutral", "Positive", "Very Positive"
2. **Recommendation Engine**: Provides product recommendations using collaborative filtering.
3. **Visualization**: Displays the sentiment distribution of the recommended products.

---

## Dataset

The project uses the **Amazon Fine Food Reviews** dataset, which contains:

- **User ID** (`UserId`)  
- **Product ID** (`ProductId`)  
- **Review Text** (`Text`)  
- **Rating** (`Score`)  

### Sample Data Format

| user_id        | product_id  | review_text                                                                                                    | rating |
|----------------|-------------|------------------------------------------------------------------------------------------------------------------|--------|
| A3SGXH7AUHU8GW| B001E4KFG0  | I have bought several of the Vitality canned dog food products and have found them all to be of good quality.  | 5      |

---

## Installation

### Prerequisites

- **Python 3.8+**
- **Streamlit**
- **Pandas**
- **Matplotlib**
- **Scikit-Learn**

### Install Dependencies
This project uses transformer.

Run the following command to install required libraries:
   ```bash
   pip install torch transformers pandas scikit-learn numpy scipy matplotlib seaborn jupyter streamlit flask 
   ```
---

## Usage

### Step 1: Run the Streamlit App

To launch the Streamlit web interface, run:

```bash
 streamlit run src/app.py
```

### Step 2: Interact with the App

1. Enter a **User ID** in the input field. Here are some sample User IDs you can use:

   - **`A1P21J0DMTVGS7`**  
   - **`A1ZKFQLHFZAEH9`**  
   - **`A2NO1TXXS9T0EE`**  

2. Click **"Get Recommendations"** to receive personalized product suggestions.

3. The app will display:

   - A list of recommended products for the entered User ID.
   - Sentiment distribution charts for each recommended product.


## Project Structure

```
pythonProject/
│-- data/
│   ├── Reviews.csv
│   ├── test.csv
│   ├── train.csv
│   └── train_with_sentiment.csv
│-- src/
│   ├── app.py
│   ├── data_preprocessing.py
│   ├── recommendation.py
│   └── sentiment_analysis.py
│-- create_sentiment.py
│-- .gitattributes
│-- LICENSE
└-- README.md
```

- **`src/app.py`**: Streamlit app interface.  
- **`data/Reviews.csv`**: Raw dataset.  
- **`data/train_with_sentiment.csv`**: Preprocessed dataset with sentiment labels.  
- **`src/data_preprocessing.py`**: Script for data cleaning and splitting.  
- **`src/recommendation.py`**: Contains the `Recommender` class for generating recommendations.  
- **`src/sentiment_analysis.py`**: Sentiment analysis class using pre-trained models.  
- **`create_sentiment.py`**: Script for generating sentiment labels.  
---

## License

This project is licensed under the MIT License.

---
