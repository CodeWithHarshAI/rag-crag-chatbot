import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

# Load model once
bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# PostgreSQL connection
def get_conn():
    return psycopg2.connect(
        dbname=os.getenv("PG_DB", "ragdb"),
        user=os.getenv("PG_USER", "raguser"),
        password=os.getenv("PG_PASS", "ragpass"),
        host=os.getenv("PG_HOST", "localhost"),
        port=os.getenv("PG_PORT", "5432")
    )

# Ensure conversation table exists
def ensure_conversation_table():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            user_id TEXT,
            turn INTEGER,
            role TEXT,
            message TEXT,
            PRIMARY KEY (user_id, turn, role)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

# Load recent conversation context
def load_conversation_context(user_id, max_turns=5):
    ensure_conversation_table()
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        """
        SELECT turn, role, message FROM conversations
        WHERE user_id = %s
        ORDER BY turn ASC;
        """, (user_id,)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Filter to last N turns
    turns = {}
    for row in rows:
        turn = row["turn"]
        if turn not in turns:
            turns[turn] = {}
        turns[turn][row["role"]] = row["message"]

    sorted_turns = sorted(turns.items())[-max_turns:]
    context = []
    for _, msgs in sorted_turns:
        if "user" in msgs:
            context.append(f"User: {msgs['user']}")
        if "bot" in msgs:
            context.append(f"Bot: {msgs['bot']}")
    return "\n".join(context)

# Store new user-bot message pair
def store_message(user_id, message):
    ensure_conversation_table()
    conn = get_conn()
    cur = conn.cursor()

    # Determine next turn number
    cur.execute("SELECT MAX(turn) FROM conversations WHERE user_id = %s", (user_id,))
    result = cur.fetchone()
    next_turn = (result[0] or 0) + 1

    cur.execute(
        "INSERT INTO conversations (user_id, turn, role, message) VALUES (%s, %s, %s, %s)",
        (user_id, next_turn, "user", message["user"])
    )
    cur.execute(
        "INSERT INTO conversations (user_id, turn, role, message) VALUES (%s, %s, %s, %s)",
        (user_id, next_turn, "bot", message["bot"])
    )
    conn.commit()
    cur.close()
    conn.close()

# Prompt constructor
def construct_prompt_crag(chunks, user_query, history_context=None):
    prompt = "You are a helpful assistant. Use the following documents and conversation context to answer the user.\n\n"
    if history_context:
        prompt += f"Conversation History:\n{history_context}\n\n"
    prompt += f"Retrieved Chunks:\n{chunks}\n\nUser Query: {user_query}\n\nAnswer:"
    return prompt

# Context-aware retrieval using BGE
def retrieve_relevant_chunks_with_context(user_query, history_context, top_k=5):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT chunk FROM chunks")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    all_chunks = [r[0] for r in rows]
    full_query = user_query + " " + history_context if history_context else user_query
    query_embedding = bge_model.encode(full_query, convert_to_tensor=True)
    chunk_embeddings = bge_model.encode(all_chunks, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
    top_indices = np.argsort(-similarities.cpu().numpy())[:top_k]
    top_chunks = [all_chunks[i] for i in top_indices]
    return "\n\n".join(top_chunks)