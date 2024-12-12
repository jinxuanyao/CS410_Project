import pandas as pd
from sklearn.model_selection import train_test_split

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

if __name__ == "__main__":
    input_path = "data/Reviews.csv"
    df = load_and_clean_data(input_path, sample_size=100)

    train, test = split_data(df)

    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    print("Sampling done, train and test sets saved in data/train.csv")
