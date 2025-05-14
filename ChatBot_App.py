# bijli_mitra_chatbot.py

import streamlit as st
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import pickle
import time
import random
from streamlit_chat import message
from PIL import Image

# ---------------- Load Model & Vector DB ----------------

@st.cache_resource

def load_model_index_data():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    with open("vector_data.pkl", "rb") as f:
        vector_store = pickle.load(f)
    df = vector_store["df"]
    index = vector_store["index"]
    return model, index, df

# ---------------- Streamlit Page Setup ----------------

st.set_page_config(page_title="⚡ Bijli Mitra - Consumer Chatbot", layout="wide")
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
    }
    .stChatMessage {
        font-size: 1.1rem;
    }
    .user-message {
        background-color: #dcf8c6;
        padding: 10px 15px;
        border-radius: 18px;
        margin: 5px 40px 5px auto;
        max-width: 80%;
        animation: fadeInUp 0.5s;
    }
    .bot-message {
        background-color: #ececec;
        padding: 10px 15px;
        border-radius: 18px;
        margin: 5px auto 5px 40px;
        max-width: 80%;
        animation: fadeInUp 0.5s;
    }
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)


st.title("⚡ Bijli Mitra - DISCOM Consumer Assistant")
st.markdown("Hello! I'm **Bijli Mitra**, your assistant for electricity-related queries. How can I help you today?")

model, index, df = load_model_index_data()

# ---------------- Chat History ----------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [("bot", "👋 Namaste! Bijli Mitra here. Asked me any thing ")]

user_input = st.chat_input("Write Your Quetion Here....")

# ---------------- Typing Animation ----------------
def simulate_typing_response(text):
    full_response = ""
    for char in text:
        full_response += char
        time.sleep(random.uniform(0.01, 0.03))
        yield full_response

# ---------------- Process Input ----------------

if user_input:
    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("Bijli Mitra soch raha hai..."):
        query_vec = model.encode([user_input])
        D, I = index.search(np.array(query_vec), k=1)
        matched_answer = df.iloc[I[0][0]]["answer"]

        bot_response_gen = simulate_typing_response(matched_answer)
        response = ""
        for part in bot_response_gen:
            response = part
        st.session_state.chat_history.append(("bot", response))

# ---------------- Display Chat ----------------

for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"<div class='user-message'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-message'>{msg}</div>", unsafe_allow_html=True)
