from src.recommender import ALSRecommender


class RecommendationService:
    def __init__(self, recommender: ALSRecommender) -> None:
        self.recommender = recommender

    def recommend(self, user_id: int | str, top_k: int = 10) -> dict:
        return self.recommender.recommend(user_id=user_id, top_k=top_k)