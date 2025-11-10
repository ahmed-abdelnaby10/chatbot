import os
import re
import streamlit as st
from openai import OpenAI

# =========================
# 1) API KEY SETUP
# =========================

HUGGINGFACE_API_KEY = st.secrets.get(
    "HUGGINGFACE_API_KEY", os.getenv("HUGGINGFACE_API_KEY", "")
)

if not HUGGINGFACE_API_KEY:
    st.error(
        "❌ Missing HUGGINGFACE_API_KEY. Set it as an environment variable or Streamlit secret."
    )
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
    "<h1 style='text-align:center;'>🎯 MY AI APPLICATION</h1>",
    unsafe_allow_html=True,
)
st.write("---")

# =========================
# 3) AVAILABLE MODELS
# =========================

MODELS = {
    "Qwen/Qwen3-32B": "Qwen/Qwen3-32B",
    "Qwen/Qwen3-Next-80B-A3B-Instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "google/gemma-2-2b-it": "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
}

# =========================
# 4) HELPERS
# =========================


def strip_think_blocks(text: str) -> str:
    """
    Remove any <think>...</think> blocks from the response.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def estimate_tokens(text: str) -> int:
    """
    Very rough token estimate (good enough for display).
    """
    if not text:
        return 0
    return len(text.split())


# =========================
# 5) SIDEBAR: MODEL + SYSTEM PROMPT
# =========================

st.sidebar.title("Settings")

selected_model = st.sidebar.selectbox(
    "Choose model",
    options=list(MODELS.keys()),
    index=0,
)

# Default system prompt depends on model (Arabic-friendly for Qwen2.5-7B)
default_prompt = (
    "You are a helpful AI assistant. You understand Arabic and Egyptian dialect (العامية المصرية) very well."
    if selected_model == "Qwen/Qwen2.5-7B-Instruct"
    else "You are a helpful AI assistant."
)

system_prompt = st.sidebar.text_area(
    "System prompt",
    value=default_prompt,
    help="Controls how the model behaves.",
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

# Sync system prompt with first message
if (
    st.session_state.messages
    and st.session_state.messages[0]["role"] == "system"
    and st.session_state.messages[0]["content"] != system_prompt
):
    st.session_state.messages[0]["content"] = system_prompt

# Reset conversation button
if st.sidebar.button("Reset conversation"):
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
    st.experimental_rerun()

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
# 7) USER INPUT + STREAMING RESPONSE
# =========================

user_input = st.chat_input("Type your message...")

if user_input:
    # 1) Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2) Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # 3) Prepare assistant message container
    full_reply = ""
    clean_reply = ""
    output_tokens = 0

    with st.chat_message("assistant"):
        placeholder = st.empty()
        token_box = st.empty()

        try:
            # Streaming call
            stream = client.chat.completions.create(
                model=MODELS[selected_model],
                messages=st.session_state.messages,
                max_tokens=400,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    full_reply += delta.content

                    # Clean on the fly (removing <think> if present)
                    clean_reply = strip_think_blocks(full_reply)

                    # Update typing effect
                    placeholder.markdown(clean_reply)

                    # Update token estimate
                    output_tokens = estimate_tokens(clean_reply)
                    token_box.caption(f"Output tokens (approx): {output_tokens}")

        except Exception as e:
            clean_reply = f"⚠️ Error from model: {e}"
            placeholder.markdown(clean_reply)
            output_tokens = estimate_tokens(clean_reply)
            token_box.caption(f"Output tokens (approx): {output_tokens}")

    # 4) Save final clean reply in history
    st.session_state.messages.append({"role": "assistant", "content": clean_reply})
