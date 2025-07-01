##The Streamlit Web App

import streamlit as st
from pdf_loader import extract_text_from_pdf
from chain import get_chain

st.set_page_config(page_title="PDF ChatBot with Ollama", layout="wide")
st.title("📄 Chat with your PDF (Local LLM via Ollama)")

pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf:
    with st.spinner("Reading your PDF..."):
        text = extract_text_from_pdf(pdf)
        st.success("✅ PDF content extracted!")

    question = st.text_input("Ask a question about the PDF")

    if question:
        chain = get_chain()
        with st.spinner("Thinking..."):
            answer = chain.run({'context': text, 'question': question})
            st.write("### 🤖 Answer:")
            st.write(answer)
