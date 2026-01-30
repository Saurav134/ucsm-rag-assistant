import streamlit as st
import requests
import os
import uuid

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

# --- Config (Secrets first, then env) ---
API_BASE = os.getenv("RAG_API_URL", "").strip().rstrip("/")
HF_TOKEN = os.getenv("HF_TOKEN", "")

if not API_BASE:
    st.error(
        "RAG_API_URL is not set. Set it in Space Secrets (recommended) or as an environment variable, "
        "e.g., https://<backend>.hf.space"
    )
    st.stop()

QUERY_URL = f"{API_BASE}/query/"
WARMUP_URL = f"{API_BASE}/warmup"
HEALTH_URL = f"{API_BASE}/"

# Add auth header for private backend Spaces
headers = {}
if HF_TOKEN:
    headers["Authorization"] = f"Bearer {HF_TOKEN}"

# --- Session ID (restore) ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

with st.sidebar:
    st.markdown("---")
    st.caption(f"Session: `{st.session_state.session_id}`")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Health"):
            try:
                r = requests.get(HEALTH_URL, headers=headers, timeout=20)
                # backend may return json; if not, show text
                try:
                    st.write(r.json())
                except Exception:
                    st.write(r.text)
            except Exception as e:
                st.error(f"Health check failed: {e}")

    with col2:
        if st.button("Warmup"):
            try:
                r = requests.get(WARMUP_URL, headers=headers, timeout=600)
                if r.status_code == 200:
                    st.success("Warmup complete.")
                else:
                    st.warning(f"Warmup returned {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"Warmup failed: {e}")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(msg)

user_input = st.chat_input("Ask anything about UCS Manager (CLI or GUI)...")

if user_input:
    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            placeholder = st.empty()
            final_text = ""

            params = {
                "query": user_input,
                "session_id": st.session_state.session_id,  # send session_id to backend
            }

            try:
                with requests.get(
                    QUERY_URL,
                    params=params,
                    headers=headers,
                    stream=True,
                    timeout=600
                ) as r:

                    if r.status_code == 503:
                        try:
                            detail = r.json().get("detail", r.text)
                        except Exception:
                            detail = r.text
                        final_text = f"Backend not ready: {detail}"
                        placeholder.markdown(final_text)

                    elif r.status_code != 200:
                        final_text = (
                            f"Backend error ({r.status_code}). "
                            f"If the backend Space is private, ensure HF_TOKEN is set correctly.\n\n{r.text}"
                        )
                        placeholder.markdown(final_text)

                    else:
                        for chunk in r.iter_content(chunk_size=1024):
                            if chunk:
                                text = chunk.decode("utf-8", errors="ignore")
                                final_text += text
                                placeholder.markdown(final_text)

            except Exception as e:
                final_text = f"Error: {e}"
                placeholder.markdown(final_text)

    st.session_state.messages.append(("assistant", final_text))
