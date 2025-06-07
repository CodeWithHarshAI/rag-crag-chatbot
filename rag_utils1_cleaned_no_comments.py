






























        

















import numpy as np
import os
import hashlib
from threading import Lock
from dotenv import load_dotenv
import openai
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_values

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
db_write_lock = Lock()

def get_conn():
    return psycopg2.connect(
        dbname=os.getenv("PG_DB", "ragdb"),
        user=os.getenv("PG_USER", "raguser"),
        password=os.getenv("PG_PASS", "ragpass"),
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5432")
    )

def chunk_text(text, size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
    return chunks

def store_chunks(doc_name, full_text, use_openai=False, progress_callback=None):
    with db_write_lock:
        conn = get_conn()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_name TEXT,
                chunk TEXT,
                embedding BYTEA,
                embedding_model TEXT
            );
        """)
        conn.commit()

        chunks = chunk_text(full_text)
        total_chunks = len(chunks)
        print(f"[📄] Processing `{doc_name}` → {total_chunks} chunks")

        embedding_model ="openai" if use_openai else "bge"
        for i, chunk in enumerate(chunks):
            if use_openai:
                print(f"[🔑] OpenAI embedding for chunk {i+1}/{total_chunks}")
                embedding = openai.Embedding.create(
                    model="text-embedding-3-small",
                    input=chunk
                )["data"][0]["embedding"]
                embedding = np.array(embedding, dtype=np.float32).tobytes()
            else:
                print(f"[🧠] BGE embedding for chunk {i+1}/{total_chunks}")
                embedding = bge_model.encode(chunk).astype(np.float32).tobytes()

            chunk_id = hashlib.md5((doc_name + chunk).encode()).hexdigest()

            cursor.execute("""
                INSERT INTO chunks (id, doc_name, chunk, embedding, embedding_model)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    chunk = EXCLUDED.chunk,
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model;
            """, (chunk_id, doc_name, chunk, embedding, embedding_model))

            if progress_callback:
                progress_callback((i + 1) / total_chunks)

        conn.commit()
        conn.close()
        print(f"[✅] Finished storing embeddings for `{doc_name}`")

def retrieve_relevant_chunks(query, doc_names=None, top_k=5, use_openai=False):

    embedding_model = "openai" if use_openai else "bge"
    if doc_names is not None and len(doc_names) == 0:
        return "", []

    conn = get_conn()
    cursor = conn.cursor()


    expected_model = "openai" if use_openai else "bge"

    if doc_names:
        placeholders = ",".join(["%s"] * len(doc_names))
        cursor.execute(f"""
            SELECT doc_name, chunk, embedding FROM chunks
            WHERE doc_name IN ({placeholders}) AND embedding_model = %s;
        """, (*doc_names, expected_model))
    else:
        cursor.execute("""
            SELECT doc_name, chunk, embedding FROM chunks
            WHERE embedding_model = %s;
        """, (expected_model,))


    rows = cursor.fetchall()

    if use_openai:
        query_vec = openai.Embedding.create(
            model="text-embedding-3-small",
            input=query
        )["data"][0]["embedding"]
        query_vec = np.array(query_vec, dtype=np.float32)
    else:
        query_vec = bge_model.encode("query: " + query).astype(np.float32)

    similarities = []
    for doc_name, chunk, embedding_blob in rows:
        emb = np.frombuffer(embedding_blob, dtype=np.float32)
        sim = np.dot(query_vec, emb) / (np.linalg.norm(query_vec) * np.linalg.norm(emb))
        similarities.append((sim, doc_name, chunk))

    similarities.sort(reverse=True)
    top_chunks = similarities[:top_k]
    conn.close()

    combined_context = "\n\n".join(
        f"--- Source: {doc_name} ---\n{chunk}" for _, doc_name, chunk in top_chunks
    )
    raw_chunks = [{"doc_name": doc_name, "chunk": chunk} for _, doc_name, chunk in top_chunks]
    return combined_context, raw_chunks