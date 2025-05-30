# import pdfplumber
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.llms import Ollama
# from langchain.chains import RetrievalQA

# # 1. Extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             if page.extract_text():
#                 text += page.extract_text() + "\n"
#     return text

# # 2. Create a vector database
# def create_vector_store(text):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     docs = splitter.create_documents([text])

#     embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vectorstore = Chroma.from_documents(docs, embedding)
#     return vectorstore

# # 3. Load LLaMA via Ollama and setup QA chain
# def create_qa_chain(vectorstore):
#     llm = Ollama(model="llama3")  # make sure Ollama is running
#     retriever = vectorstore.as_retriever()
#     qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
#     return qa

# # 4. Main chatbot function
# def chatbot(pdf_path, question):
#     print("Reading PDF and setting up AI model...")
#     text = extract_text_from_pdf(pdf_path)
#     vectorstore = create_vector_store(text)
#     qa_chain = create_qa_chain(vectorstore)

#     print("Answering your question...")
#     return qa_chain.run(question)

# # Example usage
# if __name__ == "__main__":
#     pdf_path = input("Enter path to PDF file: ")
#     question = input("Enter your question: ")
#     answer = chatbot(pdf_path, question)
#     print("\nAnswer:\n", answer)


import os
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


# # 1. Extract text from PDF
def load_pdf(path):
    if not os.path.exists(path):
        raise FileNotFoundError("The specified PDF file does not exist.")
    loader = PyPDFLoader(path)
    return loader.load()

# # 2. Create a vector database
def create_vectorstore(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# # 3. Load LLaMA via Ollama and setup QA chain
def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = Ollama(model="llama3.2")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain


# # 4. Main chatbot function
def chatbot(pdf_path, question):
    try:
        pages = load_pdf(pdf_path)
        vectorstore = create_vectorstore(pages)
        qa_chain = create_qa_chain(vectorstore)
        result = qa_chain.invoke({"query": question})  # ✅ Updated line
        return result["result"]
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    pdf_path = input("Enter path to PDF file: ").strip()
    question = input("Enter your question: ").strip()
    answer = chatbot(pdf_path, question)
    print("\nAnswer:", answer)
