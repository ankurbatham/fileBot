# streamlit_app.py
import streamlit as st
import os
from pdf_bot import chatbot

st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 Chat with PDF using LLaMA 3 + LangChain")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    pdf_path = os.path.join(temp_dir, uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ PDF uploaded successfully!")
    question = st.text_input("Ask a question about the PDF:")

    if question and st.button("Ask"):
        with st.spinner("⏳ Generating answer..."):
            try:
                answer = chatbot(pdf_path, question)
                st.markdown(f"### 🧠 Answer:\n{answer}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
