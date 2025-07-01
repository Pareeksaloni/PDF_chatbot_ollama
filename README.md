# PDF_chatbot_ollama
It's a PDF chatbot which can answer any ques related to the PDF.


###Let me give you the overview of this project:

Let's walk through the **complete breakdown** of your project — a **PDF chatbot using Streamlit, LangChain, and Ollama** — including the libraries used, architecture, steps, and important details to fully understand and manage it.

---

## 🔧 Project Overview

You’ve built a **PDF Chatbot** that lets users upload a PDF and ask questions about its content using a local LLM (e.g., TinyLLaMA) via Ollama. This is powered by:

* **Streamlit**: Web UI framework.
* **LangChain**: Framework to chain LLMs with tools like retrievers and memory.
* **Ollama**: Local LLM runtime (TinyLLaMA or other models).
* **PyMuPDF / fitz**: Extract text from PDF files.

---

## 📦 Libraries Used & What They Do

| Library                                         | Role                                                               |
| ----------------------------------------------- | ------------------------------------------------------------------ |
| `streamlit`                                     | Create the web interface for file upload and chat                  |
| `langchain`                                     | Create chains of LLM calls with context (retrievers, chains)       |
| `langchain_community.llms.ollama`               | LangChain wrapper to connect to Ollama models                      |
| `PyMuPDF (fitz)`                                | Extracts text from PDF documents                                   |
| `tiktoken` or `langchain.text_splitter`         | Token-based text chunking                                          |
| `faiss` / `chromadb` / `langchain.vectorstores` | Store vector embeddings for semantic search (if using vectorstore) |
| `ollama` (CLI tool)                             | Downloads and serves local models like TinyLLaMA                   |
| `dotenv` / `os`                                 | Load secrets, if needed                                            |
| `typing`                                        | Type hinting for better code clarity                               |

---

## 🧠 Architecture Diagram

```
       ┌────────────┐
       │  Streamlit │   ← User Uploads PDF / Asks Question
       └────┬───────┘
            │
            ▼
 ┌─────────────────────┐
 │  Extract Text (fitz)│
 └─────────────────────┘
            │
            ▼
 ┌─────────────────────────────┐
 │  Split Text into Chunks     │
 │ (LangChain TextSplitter)    │
 └─────────────────────────────┘
            │
            ▼
 ┌────────────────────────────────────────┐
 │ Convert Chunks → Embeddings (Optional)│
 │ Store in Vector DB (FAISS / Chroma)   │
 └────────────────────────────────────────┘
            │
            ▼
 ┌────────────────────────────────┐
 │ Ask LLM via LangChain → Ollama│ ← using TinyLLaMA
 └────────────────────────────────┘
            │
            ▼
       ┌────────────┐
       │   Answer   │
       └────────────┘
```

---

## 🧱 Step-by-Step Breakdown

### 1. **Set Up Ollama and Pull the Model**

```bash
ollama pull tinyllama
ollama run tinyllama
```

This runs a local REST API server on `localhost:11434`.

---

### 2. **Install Python Dependencies**

```bash
pip install streamlit langchain langchain-community PyMuPDF
```

Optionally, install FAISS if you’re using vector search:

```bash
pip install faiss-cpu
```

---

### 3. **Extract Text from PDF**

Using PyMuPDF:

```python
import fitz

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text
```

---

### 4. **Split Text into Chunks**

This helps the model handle long documents:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = splitter.create_documents([full_text])
```

---

### 5. **Vector Embedding and Search (Optional)**

If you're using semantic search:

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings(model='tinyllama')
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()
```

---

### 6. **LangChain LLM Setup with Ollama**

```python
from langchain_community.llms import Ollama

llm = Ollama(model="tinyllama")  # Make sure model is pulled
```

---

### 7. **LangChain QA Chain**

```python
from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(llm, chain_type="stuff")  # or "map_reduce" for larger docs
answer = chain.run({'context': text, 'question': user_input})
```

---

### 8. **Streamlit App UI**

```python
import streamlit as st

st.title("PDF Chatbot with TinyLLaMA")

pdf = st.file_uploader("Upload a PDF", type="pdf")
if pdf:
    text = extract_text_from_pdf(pdf)
    question = st.text_input("Ask a question about the PDF")
    if question:
        answer = chain.run({'context': text, 'question': question})
        st.write("Answer:", answer)
```

---

## 🧪 Testing & Debugging Tips

* ❗ **404 error** from Ollama? Make sure model is pulled (`ollama pull tinyllama`)
* 🧠 **Too long context error**? Reduce chunk size or use summarization
* 💥 **TypeError / unhashable list**? Check your `List` usage, especially in `middlewares`

---

## 🚀 Enhancements You Can Add Later

| Feature       | Description                        |
| ------------- | ---------------------------------- |
| Memory        | Use conversation history           |
| Summarization | Add auto-summarize button          |
| PDF Search    | Add search for keywords            |
| Multiple PDFs | Let user chat across documents     |
| UI polish     | Add sidebar, spinner, themes, etc. |

---

## 📁 Folder Structure (Suggested)

```
pdf-chatbot/
├── app.py
├── requirements.txt
├── ollama-models/
│   └── pulled models (local)
├── utils/
│   └── pdf_utils.py (extraction etc.)
```

---

## ✅ Final Thoughts

I’ve built a **local privacy-safe chatbot** that can read any document and answer questions — powered by lightweight models like TinyLLaMA, without needing OpenAI API keys.

