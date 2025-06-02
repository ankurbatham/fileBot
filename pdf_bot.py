#multiple files insert 


import os
import concurrent.futures
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Store chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Load multiple PDFs
def load_pdfs(pdf_files):
    documents = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda pdf: PyPDFLoader(pdf).load(), pdf_files))
    
    for docs in results:
        documents.extend(docs)
    return documents

# Process and split text
def process_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    return chunks

# Store embeddings & create retriever
def create_vectorstore(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore.as_retriever()

# Initialize chatbot with RAG
def chatbot_response(question, retriever):
    llm = Ollama(model="mistral")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        memory=memory
    )
    return qa_chain.run({"question": question})

# Main chatbot function with history support
def chatbot(pdf_paths, question):
    try:
        documents = load_pdfs(pdf_paths)
        chunks = process_documents(documents)
        retriever = create_vectorstore(chunks)
        response = chatbot_response(question, retriever)
        return response
    except Exception as e:
        return f"Error: {e}"

# Run chatbot interactively
if __name__ == "__main__":
    pdf_list = input("Enter paths to PDF files, separated by commas: ").strip().split(",")
    question = input("Enter your question: ").strip()
    answer = chatbot(pdf_list, question)
    print("\nAnswer:", answer)




# import os
# from langchain_community.llms import Ollama
# from langchain.chains import RetrievalQA
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter


# # # 1. Extract text from PDF
# def load_pdf(path):
#     if not os.path.exists(path):
#         raise FileNotFoundError("The specified PDF file does not exist.")
#     loader = PyPDFLoader(path)
#     return loader.load()

# # # 2. Create a vector database
# def create_vectorstore(pages):
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     texts = splitter.split_documents(pages)

#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectorstore = FAISS.from_documents(texts, embeddings)
#     return vectorstore

# # # 3. Load LLaMA via Ollama and setup QA chain
# def create_qa_chain(vectorstore):
#     retriever = vectorstore.as_retriever()
#     llm = Ollama(model="llama3.2")
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True
#     )
#     return qa_chain


# # # 4. Main chatbot function
# def chatbot(pdf_path, question):
#     try:
#         pages = load_pdf(pdf_path)
#         vectorstore = create_vectorstore(pages)
#         qa_chain = create_qa_chain(vectorstore)
#         result = qa_chain.invoke({"query": question})  # ✅ Updated line
#         return result["result"]
#     except Exception as e:
#         return f"Error: {e}"

# if __name__ == "__main__":
#     pdf_path = input("Enter path to PDF file: ").strip()
#     question = input("Enter your question: ").strip()
#     answer = chatbot(pdf_path, question)
#     print("\nAnswer:", answer)
