import re
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

from src.recommendation import RecommendationEngine
from src.rag import ProductRAG
from pathlib import Path

load_dotenv()

# ============================================================
# BACKEND
# ============================================================

recommendation_engine = RecommendationEngine()

BASE_DIR = Path(__file__).resolve().parent.parent

rag = ProductRAG(
    product_file=str(BASE_DIR / "data" / "products.csv"),
    persist_directory=str(BASE_DIR / "data" / "chroma_db")
)

# ============================================================
# TOOL 1 — PURCHASE RECOMMENDATIONS
# ============================================================

@tool
def recommend_products(stock_code: str, limit: int = 5):
    """
    Recommend products frequently bought together with a product.
    Uses Neo4j purchase relationships.
    """
    try:
        stock_code = str(stock_code).strip()
        product = rag.get_product(stock_code)

        if product is None:
            return {
                "error": f"Product {stock_code} was not found.",
                "stock_code": stock_code
            }

        results = recommendation_engine.recommend(
            stock_code=stock_code,
            limit=limit
        )

        return {
            "product": product,
            "recommendations": results or []
        }

    except Exception as e:
        print(f"Neo4j recommendation error: {e}")
        return {
            "error": (
                "The purchase recommendation service "
                "is temporarily unavailable."
            )
        }

# ============================================================
# TOOL 2 — SEMANTIC PRODUCT SEARCH
# ============================================================
@tool
def search_products(query: str, k: int = 5):
    """
    Search the product catalog using ChromaDB semantic search.

    If the query contains a stock code, first resolve it
    to the exact product. If the stock code does not exist,
    return a product-not-found message.
    """

    try:
        query = str(query).strip()
        search_query = query
        stock_code = None

        # Detect stock code
        match = re.search(r"\b\d+[A-Za-z]*\b", query)

        if match:
            stock_code = match.group(0).strip()

            # Resolve stock code
            product = rag.get_product(stock_code)

            # Stock code does not exist
            if product is None:
                return {
                    "error": f"Product with stock code {stock_code} was not found in the catalog."
                }

            # Use exact product name for semantic search
            search_query = product["name"]

        # Semantic search
        results = rag.search(
            query=search_query,
            k=k + 1
        )

        if not results:
            return {
                "query": query,
                "products": []
            }

        products = []

        for doc in results:
            code = str(
                doc.metadata.get("stock_code", "")
            ).strip()

            # Don't return original product
            if stock_code and code == stock_code:
                continue

            if not code:
                continue

            products.append({
                "stock_code": code,
                "name": str(
                    doc.metadata.get("name", "")
                ).strip()
            })

            if len(products) >= k:
                break

        return {
            "query": query,
            "products": products
        }

    except Exception as e:
        print(f"ChromaDB search error: {e}")

        return {
            "error": (
                "The product search service "
                "is temporarily unavailable."
            )
        }

# ============================================================
# TOOL 3 — HYBRID RECOMMENDATIONS
# ============================================================

@tool
def hybrid_recommend_products(
    stock_code: str,
    limit: int = 5
):
    """
    Hybrid recommendation using Neo4j purchase relationships
    and ChromaDB semantic similarity.
    """
    try:
        stock_code = str(stock_code).strip()

        # Resolve original product
        product = rag.get_product(stock_code)

        if product is None:
            return {
                "error": f"Product {stock_code} was not found."
            }

        product_name = product["name"]

        # Purchase recommendations
        purchase_recommendations = (
            recommendation_engine.recommend(
                stock_code=stock_code,
                limit=limit
            )
        )

        # Semantic recommendations
        similar_products = rag.search(
            query=product_name,
            k=limit + 3
        )

        combined = {}

        # Add purchase recommendations
        for item in purchase_recommendations:
            code = str(
                item["stock_code"]
            ).strip()

            if code == stock_code:
                continue

            combined[code] = {
                "stock_code": code,
                "name": str(
                    item.get("name", "")
                ).strip(),
                "purchase_count": item.get(
                    "purchase_count", 0
                ),
                "confidence": item.get(
                    "confidence", 0
                ),
                "lift": item.get(
                    "lift", 0
                ),
                "purchase_score": 0.0,
                "semantic_score": 0.0
            }

        # Add semantic recommendations
        semantic_rank = 0

        for doc in similar_products:
            code = str(
                doc.metadata.get(
                    "stock_code", ""
                )
            ).strip()

            if not code or code == stock_code:
                continue

            name = str(
                doc.metadata.get("name", "")
            ).strip()

            semantic_rank += 1
            semantic_score = 1 / semantic_rank

            if code not in combined:
                combined[code] = {
                    "stock_code": code,
                    "name": name,
                    "purchase_count": 0,
                    "confidence": 0,
                    "lift": 0,
                    "purchase_score": 0.0,
                    "semantic_score": semantic_score
                }
            else:
                combined[code][
                    "semantic_score"
                ] = semantic_score

        # Maximum purchase signals
        max_purchase_count = max(
            (
                item["purchase_count"]
                for item in combined.values()
            ),
            default=0
        )

        max_confidence = max(
            (
                item["confidence"]
                for item in combined.values()
            ),
            default=0
        )

        max_lift = max(
            (
                item["lift"]
                for item in combined.values()
            ),
            default=0
        )

        # Calculate purchase score
        for item in combined.values():
            count_score = (
                item["purchase_count"] / max_purchase_count
                if max_purchase_count > 0
                else 0.0
            )

            confidence_score = (
                item["confidence"] / max_confidence
                if max_confidence > 0
                else 0.0
            )

            lift_score = (
                item["lift"] / max_lift
                if max_lift > 0
                else 0.0
            )

            item["purchase_score"] = (
                0.4 * count_score
                + 0.3 * confidence_score
                + 0.3 * lift_score
            )

        # Calculate hybrid score
        for item in combined.values():
            item["hybrid_score"] = (
                0.7 * item["purchase_score"]
                + 0.3 * item["semantic_score"]
            )

        # Sort
        recommendations = sorted(
            combined.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        return {
            "original_product": product,
            "recommendations": recommendations[:limit]
        }

    except Exception as e:
        print(f"Hybrid recommendation error: {e}")
        return {
            "error": (
                "The hybrid recommendation service "
                "is temporarily unavailable."
            )
        }

# ============================================================
# LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

tools = [
    recommend_products,
    search_products,
    hybrid_recommend_products
]

llm_with_tools = llm.bind_tools(tools)

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an AI Commerce Assistant.

You help users discover products using:
1. Purchase history
2. Semantic product similarity
3. Hybrid recommendations

TOOL 1 — recommend_products

Use for:
- Frequently bought together
- Customers also bought
- Purchase history recommendations
- Cross-sell or bundle suggestions

TOOL 2 — search_products

Use for:
- Find similar products
- Search products
- Find related products
- Product catalog search

TOOL 3 — hybrid_recommend_products

Use for:
- Best recommendations
- Intelligent recommendations
- Recommendations using purchase history and similarity
- Combined recommendations

IMPORTANT:

- Never confuse similar products with frequently bought together.
- Use recommend_products for purchase-history questions.
- Use search_products for semantic similarity questions.
- Use hybrid_recommend_products for combined recommendations.

PRODUCT ID RULES:

- Stock codes identify products exactly.
- Never change a stock code.
- Never associate a stock code with a different product name.
- Always trust product information returned by the tools.
- If a tool returns original_product, use that exact stock code
  and product name.
- Never infer a product name from a semantic-search result.
- Never replace the original product with a recommended product.

TOOL DATA RULES:

- Use only information returned by the tools.
- Do not invent price, material, size, brand, color, design,
  compatibility, or other product attributes.
- Do not claim products are complementary unless the data
  supports the claim.
- Preserve exact stock codes and product names returned by tools.

CONVERSATION MEMORY:

- Use previous messages when answering follow-up questions.
- Resolve references such as:
  "it", "this product", "that product", "the same product",
  and "this one" using conversation history.
- Do not ask the user to repeat information already known.
- Keep using the same product unless the user clearly changes it.

RESPONSE RULES:

- If the request is clear, directly use the appropriate tool.
- Do not ask unnecessary clarification questions.
- Give concise and useful answers.
"""

# ============================================================
# LLM NODE
# ============================================================

def call_llm(state: MessagesState):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }

# ============================================================
# ROUTER
# ============================================================

def should_continue(state: MessagesState):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END

# ============================================================
# LANGGRAPH
# ============================================================

tool_node = ToolNode(tools)

workflow = StateGraph(MessagesState)

workflow.add_node("llm", call_llm)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "llm")

workflow.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge("tools", "llm")

# ============================================================
# MEMORY
# ============================================================

checkpointer = InMemorySaver()

app = workflow.compile(
    checkpointer=checkpointer
)

# ============================================================
# PUBLIC FUNCTION
# ============================================================

def ask_agent(
    question: str,
    thread_id: str = "user_1"
):
    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=question)
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return result["messages"][-1].content