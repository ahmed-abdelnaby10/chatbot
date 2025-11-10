import os
import streamlit as st
from openai import OpenAI

# =========================
# 1) API KEY SETUP
# =========================
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not HUGGINGFACE_API_KEY:
    st.error("❌ Missing HUGGINGFACE_API_KEY. Set it as an environment variable.")
    st.stop()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HUGGINGFACE_API_KEY,
)

# =========================
# 2) PAGE CONFIG
# =========================
st.set_page_config(
    page_title="My HF Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.markdown(
    "<h1 style='text-align:center;'>🎯 MY AI APPLICATION</h1>", unsafe_allow_html=True
)
st.write("---")

# =========================
# 3) AVAILABLE MODELS
# =========================
MODELS = {
    "Qwen/Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "google/gemma-2-2b-it": "google/gemma-2-2b-it",
}

# =========================
# 4) SIDEBAR: MODEL + SYSTEM PROMPT
# =========================
st.sidebar.title("Settings")

selected_model = st.sidebar.selectbox(
    "Choose model",
    options=list(MODELS.keys()),
    index=0,
)

system_prompt = st.sidebar.text_area(
    "System prompt",
    value="You are a helpful AI assistant.",
    help="Controls how the model behaves.",
)

if st.sidebar.button("Reset conversation"):
    st.session_state.messages = []

# =========================
# 5) INIT SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

# Sync system prompt with first message
if st.session_state.messages and st.session_state.messages[0]["role"] == "system":
    st.session_state.messages[0]["content"] = system_prompt

# =========================
# 6) RENDER CHAT HISTORY
# =========================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

# =========================
# 7) USER INPUT + CALL MODEL
# =========================
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Call HF model using OpenAI-compatible API
    try:
        response = client.chat.completions.create(
            model=MODELS[selected_model],
            messages=st.session_state.messages,
            max_tokens=400,
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"⚠️ Error from model: {e}"

    # Show reply
    with st.chat_message("assistant"):
        st.markdown(reply)

    # Save reply to history
    st.session_state.messages.append({"role": "assistant", "content": reply})
