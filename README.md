# 🛒 AI Commerce Recommendation Agent

> 🔴 **[Live Demo](https://razorpay-ai-commerce-agent-fixlfarerpdgwx99tnlkvp.streamlit.app/)**

An AI-powered e-commerce recommendation system that combines
**purchase history**, **semantic product similarity**, **Neo4j**,
**ChromaDB**, **LangGraph**, and **Groq LLMs** to provide intelligent
product recommendations through a natural-language interface.

## 🚀 Features

- 🔎 Semantic product search using ChromaDB
- 🛍️ Frequently-bought-together recommendations using Neo4j
- 🤖 Hybrid recommendations using purchase history + semantic similarity
- 🧠 AI agent powered by LangGraph
- 💬 Natural-language interaction using Streamlit
- 🗃️ Product catalog search using vector embeddings
- 🔗 Graph-based purchase relationships
- 💾 Conversation memory using LangGraph
- ⚡ Fast dependency management using uv

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
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
          ChromaDB       Neo4j       Hybrid Engine
              │            │            │
              ▼            ▼            ▼
        Semantic Search  Purchase   Combined Ranking
                         History
              │            │            │
              └────────────┼────────────┘
                           ▼
                       Groq LLM
                           │
                           ▼
                         User