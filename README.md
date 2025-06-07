# 📚 Multi-Modal ChatBot: RAG + CRAG with PostgreSQL

This project is a robust, scalable conversational AI system that integrates **Retrieval-Augmented Generation (RAG)** and **Conversational RAG (CRAG)** in a clean, togglable interface using **Streamlit**, **PostgreSQL**, and **OpenAI** or **BGE embeddings**.

Built as an evolution of a production-grade internal tool, this repo now serves as a high-quality, reusable **personal project** for document-based chatbot development.

---

## 🚀 Features

### 🔍 RAG (Retrieval-Augmented Generation)
- Embeds and retrieves top relevant document chunks for each user query.
- Supports both **OpenAI** and **BGE** embeddings.
- Stateless: each query is processed independently.

### 🧠 CRAG (Conversational RAG)
- Maintains a full user-bot conversation history (stored in PostgreSQL).
- Retrieval and response generation are contextualized using recent dialogue.
- Auto-formatted history embedded into the prompt dynamically.

### 🧰 Tooling
- Vector storage and history: **PostgreSQL**
- Embedding models: `OpenAI` / `BAAI/bge-base-en-v1.5`
- Streamlit UI with RAG/CRAG toggle
- File ingestion, chunking, and OCR-ready hooks

---

## 🧱 Architecture

```text
                  ┌──────────────────────┐
                  │   Streamlit UI       │ ◄── User interaction (Chat.py)
                  └────────┬─────────────┘
                           │
             ┌─────────────▼──────────────┐
             │ RAG / CRAG Mode Selector   │ ◄── User selects mode
             └──────┬────────────┬────────┘
                    │            │
     ┌──────────────▼──┐     ┌───▼────────────────┐
     │  RAG Retriever  │     │  CRAG Retriever    │
     │ (rag_utils1.py) │     │ (crag_utils_pg.py) │
     └─────────────────┘     └────────────────────┘
             │                       │
        ┌────▼────┐            ┌─────▼──────┐
        │  Embeds │            │ History +  │
        │ Chunks  │            │ Contextual │
        └────┬────┘            │ Retrieval  │
             │                 └────────────┘
        ┌────▼──────────────────────────────┐
        │   PostgreSQL: chunks + history    │
        └───────────────────────────────────┘
```

---

## 💾 File Structure

```bash
my-chat-app/
├── Chat_with_CRAG_PostgreSQL.py   # Streamlit frontend with RAG/CRAG toggle
├── rag_utils1_cleaned_no_comments.py # RAG utilities (chunking, store, retrieve)
├── crag_utils_pg.py               # CRAG utilities with PostgreSQL backend
├── .env                           # Environment variables (API keys, DB config)
├── requirements.txt               # Python dependencies
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/my-chat-app.git
cd my-chat-app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in root:

```dotenv
OPENAI_API_KEY=your_openai_key
PG_HOST=localhost
PG_PORT=5432
PG_DB=ragdb
PG_USER=raguser
PG_PASS=ragpass
```

### 4. Run the App

```bash
streamlit run Chat_with_CRAG_PostgreSQL.py
```

---

## 🧪 Optional Enhancements

- Add login control (`auth.py`) for real user security
- Integrate OCR pipeline using `pytesseract` and `PIL`
- Serve on cloud platforms (e.g., Streamlit Cloud, Docker, GCP, AWS)

---

## 📜 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

## 🙋‍♂️ Author

Created by [Your Name] as a personal showcase of full-stack applied AI, with production-quality structure and deployment readiness.
