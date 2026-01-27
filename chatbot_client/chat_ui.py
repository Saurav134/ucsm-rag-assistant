import streamlit as st
import requests

st.set_page_config(
    page_title="UCSM CLI guide helper chatbot",
    layout="centered"
)

st.title("UCSM CLI guide helper chatbot")

API_URL = "http://rag:8000/query/"

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
user_input = st.chat_input("Ask a question about UCSM...")

if user_input:

    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
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
