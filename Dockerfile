# Use Python 3.13 as base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir \
    langchain \
    langchain-community \
    langchain-ollama \
    chromadb \
    pypdf

# Install Ollama inside the container
RUN curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama before pulling the model
RUN ollama serve & sleep 5 && ollama pull phi3

# Expose necessary ports
EXPOSE 11434

# Run chatbot script
CMD ["python", "mul_pdfs_bot.py"]