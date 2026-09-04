from src.neo4j_client import Neo4jClient


class RecommendationEngine:

    def __init__(self):
        self.neo4j = Neo4jClient()

    def recommend(self, stock_code, limit=5):

        stock_code = str(stock_code)

        recommendations = self.neo4j.get_recommendations(
            stock_code=stock_code,
            limit=limit
        )

        return recommendations

    def close(self):
        self.neo4j.close()