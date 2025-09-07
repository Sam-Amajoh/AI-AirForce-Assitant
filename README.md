# AI-AirForce-Assitant
# LLM-Based RAG Application with FastAPI Backend

This project sets up a local Retrieval-Augmented Generation (RAG) system using FastAPI, Hugging Face LLMs (like Zephyr or Mistral), FAISS for vector search, and PDF ingestion.

## 🧰 Requirements

Ensure you have `pip` installed and run the following to install dependencies:
```
pip install -r requirements_full.txt
```

## 📁 Setup Instructions

1. Place all source code files in the same directory.
2. Ensure your machine has enough resources (e.g., GPU or CPU with sufficient RAM) to run a 7B parameter model. The tests for this model were ran using a 7900 xtx which gave us run times between 5-15 mins to recieve basic query results back. Please bear this in mind.
3. Make sure the `.env` file is present in the working directory. It should contain your Hugging Face token.


## ✅ What You Need in `.env`

Example:
```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

## 🚀 Running the Project



To run the backend:
```
uvicorn main:app --reload
```

Visit the API docs at:
```
http://127.0.0.1:8000/docs
```
