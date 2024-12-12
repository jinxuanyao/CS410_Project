import pandas as pd
from src.sentiment_analysis import SentimentAnalyzer

if __name__ == '__main__':
    sentiment_mapping = {
        "Very Negative": 1,
        "Negative": 2,
        "Neutral": 3,
        "Positive": 4,
        "Very Positive": 5
    }
    sa = SentimentAnalyzer()
    train = pd.read_csv("data/train.csv")
    train['sentiment'] = train['review_text'].apply(sa.predict)
    train['sentiment_score'] = train['sentiment'].apply(lambda x: sentiment_mapping[x])
    train.to_csv("data/train_with_sentiment.csv", index=False)
