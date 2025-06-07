import os
from dotenv import load_dotenv
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Verify API Key
if not GROQ_API_KEY:
    st.error("Groq API Key not found. Ensure it's set in .env file.")
    st.stop()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def calculate_similarity(resume_text, job_description):
    embeddings = model.encode([resume_text, job_description])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return score * 100  # Convert to percentage

# Initialize Groq LLM
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    api_key=GROQ_API_KEY
)

# Define prompt
prompt = ChatPromptTemplate.from_template(
    "You are an AI that evaluates resumes against job descriptions.\nResume: {resume_text}\nJob Description: {job_description}"
)

# Create LangChain pipeline
chain = prompt | llm

def match_resume(resume_text, job_description):
    return chain.invoke({"resume_text": resume_text, "job_description": job_description})

# Streamlit UI
st.title("AI-Powered Resume Evaluation")

# User inputs job description
job_description = st.text_area("Enter Job Description:", value="We are looking for a Python Developer with experience in Docker, LangChain, and AI model deployment.\nSkills required: Python, Docker, LangChain, AI, NLP.")

# User uploads resume PDF
uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)

    # Compute similarity score
    similarity_score = calculate_similarity(resume_text, job_description)

    # Get AI match response
    ai_match_evaluation = match_resume(resume_text, job_description)

    # Generate remark based on score
    remark = "Suitable for the job" if similarity_score > 65 else "Not a good match"

    # Display evaluation results only
    st.subheader("Resume Evaluation Results")
    st.write(f"**Resume Match Score:** {similarity_score:.2f}%")
    st.write(f"**Remark:** {remark}")

