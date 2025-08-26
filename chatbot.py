import os
import streamlit as st
from openai import OpenAI

# Load OpenAI API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set Streamlit page config
st.set_page_config(page_title="AlphaGPT", page_icon="📈")

st.title("📈 AlphaGPT")
st.write("Ask about today’s trading signals, ETF insights, and more.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    selected_model = st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo"])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a financial assistant providing ETF and stock insights."}
    ]

# Display past messages
for msg in st.session_state["messages"][1:]:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Handle user input
if user_input := st.chat_input("Ask a question..."):
    st.chat_message("user").markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=st.session_state["messages"]
        )
        reply = response.choices[0].message.content
        st.chat_message("assistant").markdown(reply)
        st.session_state["messages"].append({"role": "assistant", "content": reply})

    except Exception as e:
        st.error(f"Error: {e}")
