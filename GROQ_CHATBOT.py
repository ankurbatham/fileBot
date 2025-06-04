import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq.chat_models import ChatGroq  # ✅ Use Groq's API

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")  # Fetch API key securely

# Store chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Load multiple PDFs sequentially
def load_pdfs(pdf_files):
    documents = []
    for pdf in pdf_files:
        if os.path.exists(pdf):  # Ensure the file exists
            loader = PyPDFLoader(pdf)
            docs = loader.load()
            documents.extend(docs)
        else:
            print(f"Error: File {pdf} not found.")
    return documents

# Process and split text
def process_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    return chunks



# Store embeddings & create retriever using Mistral
def create_vectorstore(chunks):
    # embedding_model = HuggingFaceEmbeddings(model_name="mistral")  # ✅ Using Mistral embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # ✅ Works with FAISS
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore.as_retriever()



# Initialize chatbot with Mistral & Groq
def chatbot_response(question, retriever):
    
    llm = ChatGroq(
    model="mistral-saba-24b",  # ✅ Example of a valid model
    api_key=api_key
    )
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