
# **Chat with PDFs and Websites Using RAG Pipeline**

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** that enables users to interact with semi-structured data from **PDF files** and structured/unstructured content from **websites**. The system uses text extraction, chunking, embeddings, and a vector database for efficient retrieval and response generation.

## **Features**

1. **Chat with PDFs**:
   - Extracts and processes text, tables, and semi-structured data from PDF files.
   - Allows users to query specific details (e.g., unemployment rates, tabular data).
   - Handles comparison queries (e.g., comparing fields across multiple PDFs).

2. **Chat with Websites**:
   - Crawls and scrapes content from target websites.
   - Allows natural language queries for structured/unstructured web data.
   - Ensures accurate and context-rich responses using an LLM.

3. **RAG Pipeline**:
   - **Text Chunking**: Segments extracted content into manageable chunks.
   - **Embeddings**: Converts chunks and queries into embeddings for similarity search.
   - **Vector Search**: Retrieves the most relevant content using a vector database.
   - **Response Generation**: Generates accurate, fact-based answers using an LLM.

---

## **Tech Stack**

- **Backend**: Flask, FastAPI
- **LLM**: OpenAI GPT (or any preferred LLM)
- **Embeddings**: OpenAI Embeddings, Sentence Transformers
- **Vector Database**: Pinecone, ChromaDB, FAISS, or Weaviate
- **PDF Processing**: PyPDF2, PDFMiner, PyMuPDF
- **Web Scraping**: BeautifulSoup, Scrapy, Playwright
- **Orchestration**: LangChain, LlamaIndex
- **Utilities**: Pandas, NumPy, dotenv

---

## **Setup Instructions**

### **1. Prerequisites**

- **Python** (>=3.9)  
- API keys for:
   - OpenAI (for LLM and embeddings)
   - Vector Database (Pinecone, ChromaDB, or other)  

---

### **2. Clone the Repository**

```bash
git clone https://github.com/teja85053?tab=repositories/chat-with-pdfs-and-websites.git
cd chat-with-pdfs-and-websites
```

---

### **3. Install Dependencies**

Use the provided `requirements.txt` file to install all necessary libraries:

```bash
pip install -r requirements.txt
