import streamlit as st
from src.agent import ask_agent

st.set_page_config(
    page_title="AI Commerce Assistant",
    page_icon="🛒",
    layout="centered"
)

# Sidebar
with st.sidebar:
    st.title("🛒 AI Commerce Assistant")

    st.subheader("How to use")
    st.write(
        "Ask questions about products using a stock code "
        "or product name."
    )

    st.subheader("Example queries")

    st.markdown("""
    **🔍 Similar Products**
    
    `Find products similar to 22697`

    **🛍️ Frequently Bought Together**
    
    `What products are frequently bought with 22697?`

    **⭐ Best Recommendations**
    
    `Give me the best recommendations for 22697`

    **❌ Invalid Product**
    
    `Find products similar to 999999`
    """)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = "user_1"
        st.rerun()

# Main page
st.title("🛒 AI Commerce Assistant")

st.caption(
    "Find similar products, discover frequently bought items, "
    "and get intelligent recommendations."
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_1"

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
question = st.chat_input(
    "Ask about a product..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Finding the best recommendations..."):
            try:
                response = ask_agent(
                    question=question,
                    thread_id=st.session_state.thread_id
                )

                st.markdown(response)

            except Exception as e:
                response = (
                    "Sorry, something went wrong. "
                    "Please try again."
                )

                st.error(response)
                print(f"Agent error: {e}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })