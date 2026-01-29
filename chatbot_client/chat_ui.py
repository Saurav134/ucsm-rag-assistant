import streamlit as st
import requests
import os

# -----------------------
# Config
# -----------------------
st.set_page_config(
    page_title="UCSM AI Assistant",
    layout="wide",
    page_icon="🤖"
)

st.markdown(
    """
    <h1 style="text-align:center;">🤖 UCSM AI Assistant</h1>
    <p style="text-align:center;">
    Ask questions about Cisco UCS Manager CLI & GUI guides
    </p>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("About")
    st.write(
        """
        This assistant answers questions from Cisco UCS Manager
        CLI and GUI documentation (Release 6.0).

        - 11+ GUI Guides  
        - 8+ CLI Guides  
        """
    )

    st.markdown("---")
    st.write("Built by Saurav")

# -----------------------
# Backend API URL
# -----------------------
API_URL = os.getenv("RAG_API_URL", "http://rag:8000/query/")

# -----------------------
# Session State
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------
# Render History
# -----------------------
for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(msg)

# -----------------------
# Chat Input
# -----------------------
user_input = st.chat_input("Ask anything about UCS Manager (CLI or GUI)...")

if user_input:

    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            placeholder = st.empty()
            final_text = ""

            try:
                with requests.get(
                    API_URL,
                    params={"query": user_input},
                    stream=True,
                    timeout=120
                ) as r:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            text = chunk.decode("utf-8")
                            final_text += text
                            placeholder.markdown(final_text)
            except Exception as e:
                final_text = f"Error: {e}"
                placeholder.markdown(final_text)

    st.session_state.messages.append(("assistant", final_text))
