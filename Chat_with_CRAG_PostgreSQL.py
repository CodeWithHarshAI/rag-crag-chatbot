import streamlit as st
from crag_utils_pg import load_conversation_context, store_message, construct_prompt_crag, retrieve_relevant_chunks_with_context
st.set_page_config(page_title="My Chat App", page_icon="_Learning_newcolorai_White.png", layout="wide")
import requests
import json
import os
import base64
import fitz  # PyMuPDF
import docx
import re
import time
import sqlite3
from rag_utils1 import store_chunks, retrieve_relevant_chunks
from collections import defaultdict
import pytesseract
from PIL import Image
import io

import openai
from dotenv import load_dotenv
import os
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from auth import require_login, log_user_activity
current_user = require_login()

mode = st.radio("Select Chat Mode:", ["RAG", "CRAG"])
  # Enforces login
log_user_activity("Visited", "User entered My Chat App app")


OLLAMA_API_URL = "http://localhost:11434/api/generate"

HISTORY_DIR_BASE = "chat_histories2"
current_user = require_login()

mode = st.radio("Select Chat Mode:", ["RAG", "CRAG"])

user_dir = os.path.join(HISTORY_DIR_BASE, current_user.replace("@", "_at_"))  # Sanitize for filesystem
os.makedirs(user_dir, exist_ok=True)
trash_dir = os.path.join(HISTORY_DIR_BASE, "_trash", current_user.replace("@", "_at_"))
os.makedirs(trash_dir, exist_ok=True)

UPLOAD_DIR_BASE = "uploaded_docs"
upload_dir = os.path.join(UPLOAD_DIR_BASE, current_user.replace("@", "_at_"))
os.makedirs(upload_dir, exist_ok=True)




st.markdown("""
<style>
body, .stApp {
    color: white !important;
    background-color: #0e1117 !important;
}
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
}
.stChatMessage {
    background-color: #21262d !important;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)



def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

image_path = "background.PNG"  # Replace with your image path
try:
    base64_image = get_base64_image(image_path)

    background_css = f"""
    <style>
    /* Background for the main app container */
    [data-testid="stAppViewContainer"] {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                          url("data:image/png;base64,{base64_image}");
        background-size: cover; /* Ensure the image covers the entire screen */
        background-position: center; /* Center the image */
        background-attachment: fixed; /* Keep the image fixed while scrolling */
        background-repeat: no-repeat; /* Prevent tiling */
    }}

    /* Background for the sidebar */
    [data-testid="stSidebar"] {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), 
                          url("data:image/png;base64,{base64_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}

    /* Optional: Adjust text colors for readability */
    [data-testid="stSidebar"] * {{
        color: white;
    }}

    [data-testid="stAppViewContainer"] * {{
        color: white;
    }}
    header {{
        background: transparent !important;
        box-shadow: none !important;
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)
except FileNotFoundError:
    st.error(f"Image file not found: {image_path}")

def sanitize_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def extract_text_from_file(uploaded_file):
    print(f"[📥] Extracting from: {uploaded_file.name}")
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "txt":
        try:
            return uploaded_file.read().decode("utf-8")
        except:
            return uploaded_file.read().decode("latin-1")

    elif ext == "pdf":
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            text = ""
            total_pages = len(doc)

            with st.sidebar:
                st.markdown(f"**🔍 Scanning PDF: `{uploaded_file.name}`**")
                ocr_progress = st.progress(0)

            for i, page in enumerate(doc):
                page_text = page.get_text().strip()
                if page_text:
                    text += page_text + "\n"
                else:
                    pix = page.get_pixmap(dpi=300)
                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(image)
                    text += ocr_text + "\n"

                ocr_progress.progress((i + 1) / total_pages)

            return text

    elif ext == "docx":
        doc = docx.Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)
    
    elif ext in ["png", "jpg", "jpeg"]:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(image)
    return ""

def load_chat_history(file):
    path = os.path.join(user_dir, file)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        st.session_state.document_context = data.get("document_context", "")
        st.session_state.document_name = data.get("document_name", "")
        return data.get("messages", [])
    return []









def save_chat_history():
    if "messages" in st.session_state and st.session_state.messages:
        model = st.session_state.get("model", "phi4")

        if "current_chat" not in st.session_state or not st.session_state.current_chat:
            title = generate_chat_title(st.session_state.messages, model=model)
            filename = f"{title}_{int(time.time())}.json"
            path = os.path.join(user_dir, filename)
            st.session_state.current_chat = filename
        else:
            filename = st.session_state.current_chat
            path = os.path.join(user_dir, filename)

        chat_data = {
            "messages": st.session_state.messages,
            "document_context": st.session_state.get("document_context", ""),
            "document_name": st.session_state.get("document_name", ""),
            "last_updated": time.time(),
            "chat_title": filename.replace(".json", "").replace("_", " ")
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, indent=4)





def delete_chat(file):
    import shutil  # add at the top if missing

    src = os.path.join(user_dir, file)
    dst = os.path.join(trash_dir, file)

    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"[🗑️] Moved chat to trash: {dst}")

    if st.session_state.get("current_chat") == file:
        st.session_state.messages = []
        st.session_state.current_chat = None
        st.rerun()


def reset_chat_state():
    keys_to_clear = ["messages", "document_context", "document_name", "current_chat", "selected_documents"]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


def log_context_to_terminal(context):
    separator = "\n" + "="*80 + "\n"
    print(f"{separator}🧠 DOCUMENT PREVIEW LOG:{separator}")
    doc_blocks = re.findall(r'--- START OF (.*?) ---\n(.*?)\n--- END OF \1 ---', context, re.DOTALL)
    for filename, content in doc_blocks:
        words = content.split()
        preview = " ".join(words[:100])
        print(f"\n📄 {filename} (first 100 words):\n{preview}\n")
    print(f"{separator}🧠 TOTAL CONTEXT LENGTH SENT TO OLLAMA: {len(context)} characters{separator}")

    print(f"{separator}🧠 END OF DOCUMENT PREVIEW LOG{separator}"
          
          )
    
def sanitize_filename(name):
    import re
    name = name.strip().replace("\n", " ")
    name = re.sub(r'[<>:"/\\|?*]', '', name)  # Remove illegal characters
    name = re.sub(r'\s+', '_', name)  # Replace spaces/tabs/newlines with _
    return name[:50]  # Limit length


def generate_chat_title(messages, model="phi4"):
    try:
        prompt = (
            "Generate a short, clear, unique title (max 5 words) for the following chat.\n"
            "You are strictly not allowed to use any special character while generating this also no mathematical notations"
            "The title should describe the main topic.\n\n"
            "Chat:\n" +
            "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages[-6:]])
        )

        if model.startswith("gpt"):
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a chat title generator. should be max 5 words. no special character and mathematical notations"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=16,
                temperature=0.5
            )
            title = response.choices[0].message.content.strip()

        else:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 30,
                    "temperature": 0.5,
                    "stream": False
                }
            )
            title = response.json().get("response", "").strip()

    
        safe_title = title.split("\n")[0].strip()
        safe_title = " ".join(safe_title.split()[:5])
        return sanitize_filename(safe_title or "Untitled")

    except Exception as e:
        print(f"[⚠️] Title generation failed: {e}")
        return sanitize_filename(messages[0]["content"].strip().replace(" ", "_")[:50])


with st.sidebar:
    st.image("_Learning_newcolorai_White.png")
    st.markdown("#### 🤖 Select LLM")
    model_options = {
    "SwiftMind (Fast & Light)": "phi4",
    "DeepCore (Smarter)": "qwen2.5:32b",
    "NovaMind (Most Powerful)": "gpt-4o",
    }

    selected_label = st.selectbox("LLM Model", list(model_options.keys()), index=0)
    selected_model = model_options[selected_label]

    if "model" not in st.session_state:
        st.session_state.model = selected_model
    elif st.session_state.model != selected_model:
        st.session_state.model = selected_model
        log_user_activity("ModelSwitch", f"Switched to: {selected_model}")
        st.toast(f"🔁 Switched model to `{selected_label}`", icon="⚙️")
    

    def is_openai_model():
        return st.session_state.model.startswith("gpt")

    st.header("📂 Chat History")

    def get_last_updated(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("last_updated", 0)
        except:
            return 0

    file_paths = [os.path.join(user_dir, f) for f in os.listdir(user_dir) if f.endswith(".json")]
    file_paths.sort(key=get_last_updated, reverse=True)
    files = [os.path.basename(p) for p in file_paths]

    for f in files:
        label = f.replace(".json", "").replace("_", " ")
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(label, key=f"load_{f}"):
                st.session_state.messages = load_chat_history(f)
                st.session_state.current_chat = f
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{f}"):
                delete_chat(f)

    st.markdown("---")
    if st.button("➕ New Chat"):
        log_user_activity("NewChat", "Started new chat session")

        reset_chat_state()
        st.rerun()

    uploaded_files = st.file_uploader("Upload Document", type=["txt", "pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)




    if uploaded_files:
        file_names = [f.name for f in uploaded_files]
        current_doc_name = ", ".join(file_names)

        if st.session_state.get("document_name", "") != current_doc_name:
            full_context = ""
            for uploaded_file in uploaded_files:


                uploaded_path = os.path.join(upload_dir, uploaded_file.name)
                with open(uploaded_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                text = extract_text_from_file(uploaded_file)
                with st.sidebar:
                    st.markdown(f"**🔄 Embedding: `{uploaded_file.name}`**")
                    progress_bar = st.progress(0)

                def update_progress(p):
                    progress_bar.progress(p)

                store_chunks(uploaded_file.name, text, use_openai=is_openai_model(), progress_callback=update_progress)
                log_user_activity("Upload", f"Uploaded: {uploaded_file.name}")
                full_context += f"\n\n--- START OF {uploaded_file.name} ---\n{text}\n--- END OF {uploaded_file.name} ---\n"

            st.session_state.document_context = full_context
            st.session_state.document_name = current_doc_name
            st.toast(f"📄 Uploaded: {current_doc_name}", icon="📂")


    st.markdown("#### 📄 Include documents for context")
    if st.session_state.get("document_name"):
        options = [name.strip() for name in st.session_state.document_name.split(",")]
        st.session_state.selected_documents = st.multiselect("Select documents to use:", options, default=options)
    else:
        st.session_state.selected_documents = []

    st.markdown("---")
    st.header("⚙️ Settings")
    if st.session_state.model == "phi4":
        max_token_limit = 2000
        default_value =500
    elif st.session_state.model == "qwen2.5:32b":  # qwen2.5:32b or others
        max_token_limit = 8000
        default_value= 1000
    elif st.session_state.model == "gpt-4o":
        max_token_limit = 8000
        default_value = 4000

    max_tokens = st.slider("Max Tokens", 100, max_token_limit, min(default_value, max_token_limit))
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

st.title("My Chat App")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_context" not in st.session_state:
    st.session_state.document_context = ""
if "document_name" not in st.session_state:
    st.session_state.document_name = ""
if "selected_documents" not in st.session_state:
    st.session_state.selected_documents = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    log_user_activity("Query", f"Prompt: {prompt}")
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_area = st.empty()

    

        










        def stream_response(prompt):
            doc_names = st.session_state.get("selected_documents", [])
            

            doc_context, retrieved_chunks = retrieve_relevant_chunks(
                prompt, doc_names=doc_names, top_k=10, use_openai=is_openai_model()
            )

            if doc_names:
                log_user_activity("DocQuery", f"Used RAG on: {', '.join(doc_names)}")


            with st.expander("📄 Retrieved Chunks", expanded=False):
                grouped_chunks = defaultdict(list)
                for chunk in retrieved_chunks:
                    grouped_chunks[chunk['doc_name']].append(chunk['chunk'])
                for doc_name, chunks in grouped_chunks.items():
                    st.markdown(f"### 📄 Document: `{doc_name}`")
                    for chunk in chunks:
                        st.code(chunk, language="markdown")

            recent_messages = st.session_state.messages[-6:]

            if prompt.strip().lower() in ["read the pdf", "read pdf", "summarize the pdf"]:
                prompt += " — summarize all key topics, ideas, and examples based on the uploaded document chunks."

            structured_chat = ""
            for m in recent_messages:
                role_prefix = "USER" if m["role"] == "user" else "ASSISTANT"
                structured_chat += f"\n{role_prefix}: {m['content']}\n"

            if st.session_state.model in ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]:
                openai_messages = [
                    {"role": "system", "content": f"""
        In General You are a helpful assistant. but when it comes to any document. 
        You are a document analysis assistant. Your job is to help the user by reading and summarizing from the document chunks provided below.

        DO NOT say that you cannot read or analyze PDFs — you are being given the actual document text.

        Always answer based on the document content, and include references to key concepts, sections, or data points when possible.

        Use LaTeX formatting for math expressions when needed.

        --- Document Context ---
        {doc_context}
        """}
                ] + [{"role": m["role"], "content": m["content"]} for m in recent_messages]

                try:
                    response = openai.ChatCompletion.create(
                        model=st.session_state.model,
                        messages=openai_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True
                    )

                    full_reply = ""
                    for chunk in response:
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        full_reply += delta
                        response_area.markdown(full_reply)

                    st.session_state.messages.append({"role": "assistant", "content": full_reply})
                    save_chat_history()

                except Exception as e:
                    response_area.error(f"OpenAI API error: {e}")
                return


            context = (
                "In General You are a helpful assistant. but when it comes to any document.\n\n"
                "You are a document analysis assistant. Your job is to help the user by reading and summarizing from the document chunks provided below.\n\n"
                "DO NOT say that you cannot read or analyze PDFs — you are being given the actual document text.\n\n"
                "Always answer based on the document content, and include references to key concepts, sections, or data points when possible.\n\n"
                "Use LaTeX for math formatting when applicable (use `$...$` for inline, `$$...$$` for block equations).\n\n"
                f"Relevant document context:\n{doc_context}\n\n"
                f"Chat history:\n{structured_chat}\n"
                "ASSISTANT:"
            )

            payload = {
                "model": st.session_state.get("model", "phi4"),
                "prompt": context,
                "stream": True,
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            try:
                response = requests.post(OLLAMA_API_URL, json=payload, stream=True)
                full_reply = ""
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            full_reply += data["response"]
                            response_area.markdown(full_reply)

                log_user_activity("DocQuery", f"AI Response: {full_reply}")
                st.session_state.messages.append({"role": "assistant", "content": full_reply})
                save_chat_history()
            except requests.exceptions.RequestException as e:
                response_area.error(f"Request failed: {e}")


        stream_response(prompt)