import os
import pandas as pd
import numpy as np
import faiss
import pickle
import argparse
from sentence_transformers import SentenceTransformer

# ------------------- Load and Format Input File -------------------

def try_read_and_format(file_path):
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            return None

        df.columns = df.columns.str.strip().str.lower()

        if "query" in df.columns and "solution" in df.columns:
            df = df.rename(columns={"query": "question", "solution": "answer"})
        elif "question" in df.columns and "answer" in df.columns:
            pass  # Already in correct format
        else:
            print("⚠️ Required columns not found. Expected 'Question' and 'Answer' or 'Query' and 'Solution'.")
            return None

        df = df.dropna(subset=["question", "answer"])
        return df[["question", "answer"]]

    except Exception as e:
        print(f"⚠️ Failed to load file {file_path}: {e}")
        return None


# ------------------- Create New Vector DB -------------------

def create_new_vector_db(file_path, pkl_path="vector_data.pkl"):
    df = try_read_and_format(file_path)
    if df is None or df.empty:
        print("❌ Failed to read or find valid data in the file.")
        return

    print(f"📦 Creating new vector DB with {len(df)} records...")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["question"].tolist())
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    with open(pkl_path, "wb") as f:
        pickle.dump({"df": df, "index": index}, f)

    print(f"✅ New Vector DB created and saved as '{pkl_path}'")


# ------------------- Append to Existing Vector DB -------------------

def append_to_vector_store(upload_path, pkl_path="vector_data.pkl"):
    if not os.path.exists(pkl_path):
        print(f"❌ Vector DB file '{pkl_path}' not found. Use --new to create one.")
        return

    with open(pkl_path, "rb") as f:
        vector_store = pickle.load(f)

    df_existing = vector_store["df"]
    index = vector_store["index"]

    new_df = try_read_and_format(upload_path)
    if new_df is None or new_df.empty:
        print("ℹ️ No valid data found in the file.")
        return

    # Remove duplicates
    new_df = new_df[~new_df["question"].isin(df_existing["question"])]
    if new_df.empty:
        print("✅ No new unique questions found to add.")
        return

    print(f"➕ Adding {len(new_df)} new Q&A pairs...")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    new_embeddings = model.encode(new_df["question"].tolist())
    index.add(np.array(new_embeddings))

    updated_df = pd.concat([df_existing, new_df], ignore_index=True)

    with open(pkl_path, "wb") as f:
        pickle.dump({"df": updated_df, "index": index}, f)

    print("✅ Vector DB updated with new entries!")


# ------------------- CLI Options -------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector DB Management Script")
    parser.add_argument("--file", type=str, help="Path to Q&A file (CSV/XLSX)", required=True)
    parser.add_argument("--new", action="store_true", help="Create new vector DB from file")

    args = parser.parse_args()

    if args.new:
        create_new_vector_db(args.file)
    else:
        append_to_vector_store(args.file)
