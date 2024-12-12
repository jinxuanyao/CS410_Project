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

        # For each recommended product, display the product ID and plot sentiment distribution
        for product_id in recs:
            st.write(f"### Product: {product_id}")

            # Filter reviews for the current product
            product_data = train[train['product_id'] == product_id]

            if not product_data.empty:
                # Plot sentiment distribution for the current product
                fig, ax = plt.subplots()
                product_data['sentiment'].value_counts().plot(kind='bar', ax=ax)
                ax.set_xlabel("Sentiment")
                ax.set_ylabel("Number of Reviews")
                ax.set_title(f"Sentiment Distribution for Product {product_id}")
                st.pyplot(fig)
            else:
                st.write("No sentiment data available for this product.")
    else:
        st.write(f"No recommendations found for User {user_id_input}.")

st.write("Enter a valid User ID and click the button to receive personalized recommendations.")
