import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# ------------------- Load and Clean CSV -------------------

def load_and_prepare(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()

    if "query" in df.columns and "solution" in df.columns:
        df = df.rename(columns={"query": "question", "solution": "answer"})
    elif not ("question" in df.columns and "answer" in df.columns):
        raise ValueError("Required columns not found: 'question' and 'answer' or 'query' and 'solution'")

    df = df.dropna(subset=["question", "answer"])
    return df[["question", "answer"]]

# ------------------- Main Process -------------------

def generate_vector_db():
    file_path = "audit_chatbot_qna.csv"
    output_pkl = "vector_data.pkl"

    df = load_and_prepare(file_path)
    print(f"✅ Loaded {len(df)} Q&A pairs.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["question"].tolist())

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    with open(output_pkl, "wb") as f:
        pickle.dump({"df": df, "index": index}, f)

    print(f"🎉 Vector DB saved to '{output_pkl}'")

if __name__ == "__main__":
    generate_vector_db()
