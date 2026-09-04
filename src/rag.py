import os
import pandas as pd

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class ProductRAG:

    def __init__(
        self,
        product_file="../data/products.csv",
        persist_directory="../data/chroma_db"
    ):

        # ==========================================
        # 1. Load products
        # ==========================================

        self.products = pd.read_csv(product_file)

        self.products = self.products.dropna(
            subset=["stock_code", "name"]
        )

        # Normalize stock codes
        self.products["stock_code"] = (
            self.products["stock_code"]
            .astype(str)
            .str.strip()
        )

        self.products["name"] = (
            self.products["name"]
            .astype(str)
            .str.strip()
        )

        # Remove duplicate stock codes
        self.products = (
            self.products
            .drop_duplicates(subset="stock_code")
            .reset_index(drop=True)
        )

        print(
            f"Loaded {len(self.products)} unique products"
        )

        # ==========================================
        # 2. Create documents
        # ==========================================

        self.documents = []

        for _, row in self.products.iterrows():

            document = Document(
                page_content=(
                    f"Product Name: {row['name']}\n"
                    f"Stock Code: {row['stock_code']}"
                ),
                metadata={
                    "stock_code": row["stock_code"],
                    "name": row["name"]
                }
            )

            self.documents.append(document)

        # ==========================================
        # 3. Embedding model
        # ==========================================

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

        # ==========================================
        # 4. ChromaDB
        # ==========================================

        self.persist_directory = persist_directory

        self.vectorstore = Chroma(
            collection_name="products",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

        # ==========================================
        # 5. Check database
        # ==========================================

        existing_count = self.vectorstore._collection.count()

        if existing_count == 0:

            print("ChromaDB is empty. Creating embeddings...")

            ids = [
                str(doc.metadata["stock_code"])
                for doc in self.documents
            ]

            self.vectorstore.add_documents(
                documents=self.documents,
                ids=ids
            )

            print(
                f"ChromaDB created with {len(ids)} products"
            )

        else:

            print(
                f"ChromaDB loaded successfully "
                f"({existing_count} products)"
            )

    # ==========================================
    # 6. Search
    # ==========================================

    def search(self, query, k=5):

        results = self.vectorstore.similarity_search(
            query,
            k=k
        )

        return results

    # ==========================================
    # 7. Get exact product
    # ==========================================

    def get_product(self, stock_code):

        stock_code = str(stock_code).strip()

        product = self.products[
            self.products["stock_code"] == stock_code
        ]

        if product.empty:
            return None

        row = product.iloc[0]

        return {
            "stock_code": str(row["stock_code"]),
            "name": str(row["name"])
        }