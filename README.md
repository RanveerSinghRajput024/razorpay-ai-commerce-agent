# 🛒 AI Commerce Recommendation Agent

An AI-powered e-commerce recommendation system that combines **purchase history**, **semantic product similarity**, **Neo4j**, **ChromaDB**, **LangGraph**, and **Groq LLMs** to provide intelligent product recommendations.

## 🚀 Features

- 🔎 Semantic product search using ChromaDB
- 🛍️ Frequently bought-together recommendations using Neo4j
- 🤖 Hybrid recommendations using purchase history + semantic similarity
- 🧠 AI agent powered by LangGraph
- 💬 Natural-language interaction using Streamlit
- 🗃️ Product catalog stored and searched using vector embeddings
- 🔗 Graph-based purchase relationships
- 💾 Conversation memory using LangGraph
- ⚡ Fast local development using `uv`

## 🏗️ Architecture

```text
                    User
                     │
                     ▼
               Streamlit UI
                     │
                     ▼
              LangGraph Agent
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       ChromaDB    Neo4j      Hybrid
          │          │          │
          ▼          ▼          ▼
      Semantic    Purchase    Combined
       Search     History     Ranking
          │          │          │
          └──────────┼──────────┘
                     ▼
              Groq LLM Response
                     │
                     ▼
                  User