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
    """Rough token estimate for display purposes."""
    if not text:
        return 0
    return len(text.split())


def clean_think(raw: str) -> str:
    """
    Hide <think> content nicely during streaming.

    - If no <think>: return all.
    - If <think> but no </think> yet: return text before <think>.
    - If both: remove the whole <think>...</think> block.
    """
    start = raw.find("<think>")
    if start == -1:
        # no think at all
        return raw

    end = raw.find("</think>", start)
    if end == -1:
        # think started, not closed yet -> hide everything from <think> onward
        return raw[:start]

    # both present -> cut out the whole block
    end += len("</think>")
    return raw[:start] + raw[end:]


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
    else "You are a helpful AI assistant."
)

system_prompt = st.sidebar.text_area(
    "System prompt",
    value=default_prompt,
    help="Controls how the model behaves.",
)

# Init / sync session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

if (
    st.session_state.messages
    and st.session_state.messages[0]["role"] == "system"
    and st.session_state.messages[0]["content"] != system_prompt
):
    st.session_state.messages[0]["content"] = system_prompt

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

    # 3) Stream assistant answer
    raw_reply = ""  # full raw stream (with think if any)
    visible_reply = ""  # cleaned version we show
    output_tokens = 0

    with st.chat_message("assistant"):
        placeholder = st.empty()
        token_box = st.empty()

        try:
            stream = client.chat.completions.create(
                model=MODELS[selected_model],
                messages=st.session_state.messages,
                max_tokens=400,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if not delta or not delta.content:
                    continue

                # Append raw content from model
                raw_reply += delta.content

                # Compute cleaned view each time
                visible_reply = clean_think(raw_reply)

                # Update UI
                placeholder.markdown(visible_reply)
                output_tokens = estimate_tokens(visible_reply)
                token_box.caption(f"Output tokens (approx): {output_tokens}")

        except Exception as e:
            visible_reply = f"⚠️ Error from model: {e}"
            placeholder.markdown(visible_reply)
            output_tokens = estimate_tokens(visible_reply)
            token_box.caption(f"Output tokens (approx): {output_tokens}")

    # 4) Save final clean reply in history
    st.session_state.messages.append({"role": "assistant", "content": visible_reply})
