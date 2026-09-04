import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()


class Neo4jClient:

    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")

        if not self.uri:
            raise ValueError("NEO4J_URI is missing from .env")

        if not self.username:
            raise ValueError("NEO4J_USERNAME is missing from .env")

        if not self.password:
            raise ValueError("NEO4J_PASSWORD is missing from .env")

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password)
        )

    def verify_connection(self):
        self.driver.verify_connectivity()
        return True

    def get_recommendations(self, stock_code, limit=5):

        query = """
        MATCH (p:Product {stock_code: $stock_code})
              -[r:BOUGHT_WITH]->
              (recommended:Product)

        RETURN
            recommended.stock_code AS stock_code,
            recommended.name AS name,
            r.purchase_count AS purchase_count,
            r.confidence AS confidence,
            r.lift AS lift

        ORDER BY r.lift DESC

        LIMIT $limit
        """

        with self.driver.session() as session:
            result = session.run(
                query,
                stock_code=str(stock_code),
                limit=limit
            )

            return [record.data() for record in result]

    def close(self):
        self.driver.close()