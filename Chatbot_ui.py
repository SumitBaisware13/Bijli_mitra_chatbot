import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import time
from datetime import datetime
import re

# ==================== HTML Tag Stripping Function =====================
def strip_html_tags(text):
    """Remove HTML tags from a string (user input sanitization)"""
    if not text:
        return ""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# ============= Custom CSS for Chat App UI (NO fixed min-height!) =============
st.set_page_config(page_title="Bijli Mitra Chat", layout="centered")
st.markdown("""
<style>
body, .main, .block-container {
    background: linear-gradient(120deg, #e0eafc 0%, #cfdef3 100%);
    font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
}
.header-bar {
    display: flex;
    align-items: center;
    background: linear-gradient(90deg, #005bea 0%, #36d1c4 100%);
    padding: 18px 20px 10px 20px;
    border-radius: 0 0 16px 16px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px #0088aa20;
}
.header-bar img {
    border-radius: 50%;
    height: 45px;
    margin-right: 14px;
    border: 2px solid #fff;
    background: #fff;
}
.header-bar .chatbot-title {
    font-size: 1.5rem;
    color: #fff;
    font-weight: 600;
    letter-spacing: 1px;
    margin-bottom: 2px;
}
.header-bar .chatbot-desc {
    font-size: 0.9rem;
    color: #eaf8fd;
}
.chat-window {
    max-width: 540px;
    margin: auto;
    background: #ffffffcc;
    border-radius: 18px;
    box-shadow: 0 8px 32px #005bea25;
    padding: 8px 0 12px 0;
    min-height: 0;
    height: auto;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
}
.message-row {
    display: flex;
    margin: 0 0 4px 0;
}
.message-row.user {
    flex-direction: row-reverse;
    justify-content: flex-end;
}
.message-row.bot {
    justify-content: flex-start;
}
.bubble {
    display: flex;
    flex-direction: column;
    max-width: 78%;
    padding: 12px 18px;
    border-radius: 1.6em;
    margin: 7px 8px 3px 8px;
    font-size: 1.1rem;
    position: relative;
    box-shadow: 0 1px 7px #005bea10;
    word-break: break-word;
    transition: background 0.2s;
}
.bubble.user {
    background: linear-gradient(120deg, #48c6ef 0%, #6f86d6 100%);
    color: #fff;
    align-items: flex-end;
}
.bubble.bot {
    background: linear-gradient(120deg, #ece9e6 0%, #ffffff 100%);
    color: #34495e;
    align-items: flex-start;
}
.avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    margin: 4px 8px;
    border: 2px solid #36d1c4;
    box-shadow: 0 2px 8px #36d1c422;
}
.avatar.user {
    margin-left: 12px;
    background: #48c6ef;
}
.avatar.bot {
    margin-right: 12px;
    background: #005bea;
}
.timestamp {
    font-size: 0.78em;
    color: #a1a7bb;
    margin-top: 2px;
    text-align: right;
}
.typing {
    margin: 8px 16px;
    font-size: 1rem;
    display: flex;
    align-items: center;
    color: #36d1c4;
}
.dot-flashing {
    margin-left: 5px;
    width: 1.3em;
    height: 0.3em;
    display: inline-block;
    position: relative;
}
.dot-flashing span {
    position: absolute;
    left: 0;
    width: 0.3em;
    height: 0.3em;
    background: #36d1c4;
    border-radius: 50%;
    animation: dotFlashing 1s infinite linear alternate;
}
.dot-flashing span:nth-child(2) { left: 0.45em; animation-delay: 0.33s; }
.dot-flashing span:nth-child(3) { left: 0.9em;  animation-delay: 0.66s; }
@keyframes dotFlashing {
    0% { opacity: 0.2; }
    50%,100% { opacity: 1; }
}
.stChatInput {
    border-radius: 18px !important;
    border: 2px solid #48c6ef !important;
}
</style>
""", unsafe_allow_html=True)

# ============= Load Model & Vector DB =============
@st.cache_resource
def load_model_index_data():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    with open("vector_data.pkl", "rb") as f:
        vector_store = pickle.load(f)
    df = vector_store["df"]
    index = vector_store["index"]
    return model, index, df

model, index, df = load_model_index_data()

# ============= Chatbot Header =============
st.markdown("""
<div class="header-bar">
    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png" alt="Chatbot" />
    <div>
        <div class="chatbot-title">Bijli Mitra</div>
        <div class="chatbot-desc">Your AI Electricity Assistant</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============= Chat State =============
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "bot",
            "msg": "👋 Namaste! I'm Bijli Mitra. Ask me anything about bills, connections, outages, and more!",
            "ts": datetime.now().strftime("%I:%M %p")
        }
    ]
if "thinking" not in st.session_state:
    st.session_state.thinking = False

# ==== CLEANUP: Remove any HTML accidentally saved previously ====
for entry in st.session_state.chat_history:
    entry["msg"] = strip_html_tags(entry["msg"])

# ============= Chat Window =============
st.markdown('<div class="chat-window">', unsafe_allow_html=True)

# ============= Display Chat Bubbles =============
for entry in st.session_state.chat_history:
    msg = entry["msg"]
    if not msg or not msg.strip():
        continue
    role, ts = entry["role"], entry["ts"]
    is_user = (role == "user")
    avatar_url = (
        "https://cdn-icons-png.flaticon.com/512/4712/4712035.png" if not is_user
        else "https://cdn-icons-png.flaticon.com/512/9131/9131546.png"
    )
    row_class = "user" if is_user else "bot"
    bubble_class = "bubble user" if is_user else "bubble bot"
    avatar_class = "avatar user" if is_user else "avatar bot"

    st.markdown(f"""
    <div class="message-row {row_class}">
        <div class="{avatar_class}"><img src="{avatar_url}" width="38"/></div>
        <div class="{bubble_class}">
            {msg}
            <div class="timestamp">{ts}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============= Typing Indicator =============
if st.session_state.thinking:
    st.markdown("""
    <div class="message-row bot">
        <div class="avatar bot"><img src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png" width="38"/></div>
        <div class="bubble bot typing">
            Bijli Mitra is typing
            <span class="dot-flashing"><span></span><span></span><span></span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ============= User Input =============
user_input = st.chat_input("Type your message and hit Enter...")

def simulate_typing_response(text, min_delay=1.0, max_delay=1.8):
    # Simulate realistic thinking time based on answer length
    total_delay = min(max_delay, max(min_delay, len(text) * 0.012))
    time.sleep(total_delay)
    return text

# ============= Handle Input & Response =============
if user_input:
    ts_now = datetime.now().strftime("%I:%M %p")
    clean_input = strip_html_tags(user_input)
    st.session_state.chat_history.append({
        "role": "user",
        "msg": clean_input,
        "ts": ts_now
    })
    st.session_state.thinking = True
    st.rerun()  # Show typing indicator

if st.session_state.thinking:
    # Only respond to the last user input
    if len(st.session_state.chat_history) >= 1 and st.session_state.chat_history[-1]["role"] == "user":
        with st.spinner("Bijli Mitra is thinking..."):
            last_input = st.session_state.chat_history[-1]["msg"]
            query_vec = model.encode([last_input])
            D, I = index.search(np.array(query_vec), k=1)
            matched_answer = df.iloc[I[0][0]]["answer"]

            # Simulate typing delay
            response = simulate_typing_response(matched_answer)
            ts_now = datetime.now().strftime("%I:%M %p")
            st.session_state.chat_history.append({
                "role": "bot",
                "msg": strip_html_tags(response),  # Clean bot message too, just in case!
                "ts": ts_now
            })
            st.session_state.thinking = False
        st.rerun()  # Show new bot message
