import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class Recommender:
    def __init__(self, interaction_data):
        self.data = interaction_data
        self.user_item_matrix = self._create_user_item_matrix()
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_index = list(self.user_item_matrix.index)

    def _create_user_item_matrix(self):
        pivot = self.data.pivot_table(index='user_id', columns='product_id', values='sentiment_score', fill_value=0)
        return pivot

    def recommend(self, user_id, top_k=2):
        if user_id not in self.user_index:
            return []
        user_idx = self.user_index.index(user_id)
        sim_scores = self.user_similarity[user_idx]

        similar_users_idx = np.argsort(sim_scores)[::-1]
        similar_users = [self.user_index[i] for i in similar_users_idx if i != user_idx]

        user_ratings = self.user_item_matrix.loc[user_id]
        user_unseen_products = user_ratings[user_ratings == 0].index

        scores = {}
        for p in user_unseen_products:
            product_scores = []
            for u in similar_users[:10]:
                val = self.user_item_matrix.loc[u, p]
                product_scores.append(val)
            scores[p] = np.mean(product_scores) if product_scores else 0

        recommended_products = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return recommended_products


if __name__ == "__main__":
    train = pd.read_csv("data/train_with_sentiment.csv")
    recommender = Recommender(train)
    print(recommender.recommend(user_id='user_7', top_k=3))
