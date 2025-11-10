import os
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


def estimate_tokens(text: str) -> int:
    """Very rough token estimate for display."""
    if not text:
        return 0
    return len(text.split())


def clean_think(raw: str) -> str:
    """
    Remove <think> ... </think> blocks safely.

    - If <think> starts and not closed yet -> hide from <think> onwards.
    - If both tags exist -> cut the whole block.
    - Supports multiple blocks.
    """
    while True:
        start = raw.find("<think>")
        if start == -1:
            return raw

        end = raw.find("</think>", start)
        if end == -1:
            # open but not closed yet: hide everything from <think> onwards
            return raw[:start]

        # remove this block and loop in case of more
        raw = raw[:start] + raw[end + len("</think>") :]


# =========================
# 5) SIDEBAR: MODEL + SYSTEM PROMPT
# =========================

st.sidebar.title("Settings")

selected_model = st.sidebar.selectbox(
    "Choose model",
    options=list(MODELS.keys()),
    index=0,
)

default_prompt = (
    "You are a helpful AI assistant. You understand Arabic and Egyptian dialect (العامية المصرية) very well."
    if selected_model == "Qwen/Qwen2.5-7B-Instruct"
    else "You are a helpful AI assistant. Provide complete, well-structured answers."
)

system_prompt = st.sidebar.text_area(
    "System prompt",
    value=default_prompt,
    help="Controls how the model behaves.",
)

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

# Keep system prompt synced with first message
if (
    st.session_state.messages
    and st.session_state.messages[0]["role"] == "system"
    and st.session_state.messages[0]["content"] != system_prompt
):
    st.session_state.messages[0]["content"] = system_prompt

# Reset button
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
    # 1) Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Prepare streaming containers
    raw_reply = ""  # raw stream (may contain <think>)
    visible_reply = ""  # cleaned, what we show
    output_tokens = 0

    with st.chat_message("assistant"):
        text_placeholder = st.empty()
        tokens_placeholder = st.empty()

        try:
            stream = client.chat.completions.create(
                model=MODELS[selected_model],
                messages=st.session_state.messages,
                max_tokens=2048,  # give it space so it doesn't cut early
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if not delta or not delta.content:
                    continue

                # accumulate raw text
                raw_reply += delta.content

                # clean <think> blocks on each update
                visible_reply = clean_think(raw_reply)

                # update UI
                text_placeholder.markdown(visible_reply)
                output_tokens = estimate_tokens(visible_reply)
                tokens_placeholder.caption(f"Output tokens (approx): {output_tokens}")

        except Exception as e:
            visible_reply = f"⚠️ Error from model: {e}"
            text_placeholder.markdown(visible_reply)
            output_tokens = estimate_tokens(visible_reply)
            tokens_placeholder.caption(f"Output tokens (approx): {output_tokens}")

    # 3) Save final clean reply
    st.session_state.messages.append({"role": "assistant", "content": visible_reply})
