# I have done it in colab its working fine here is the link
#https://colab.research.google.com/drive/1F771l1grMkqEFZ1Ad1YZTQ-_KGfsLhuD?usp=sharing

# Install required libraries
!pip install -q PyPDF2 sentence-transformers google-generativeai

import os
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
from typing import List
import google.generativeai as genai

# Set up Gemini API
genai.configure(api_key="AIzaSyBb8awMVXqbUKJMbcjd6VietECKIqebPkU")

class RAGPipeline:
    def __init__(self,
                 embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize RAG Pipeline components with Gemini
        """
        # Embedding Model
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Vector Storage
        self.embeddings = []
        self.documents = []

        # Language Model
        self.llm = genai.GenerativeModel('gemini-pro')

    def extract_pdf_text(self, pdf_path: str) -> List[dict]:
        """
        Extract text from PDF, creating document chunks
        """
        documents = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                documents.append({
                    'content': text,
                    'page': page_num + 1
                })

        return documents

    def index_documents(self, documents: List[dict]):
        """
        Convert documents to embeddings and store
        """
        embeddings = self.embedding_model.encode(
            [doc['content'] for doc in documents]
        )

        self.embeddings = embeddings
        self.documents = documents

    def retrieve_page_content(self, page_number: int) -> str:
        """
        Retrieve content from a specific page
        """
        for doc in self.documents:
            if doc['page'] == page_number:
                return doc['content']
        return "Page not found"

    def generate_response(self, query: str) -> str:
        """
        Generate a response using retrieved context
        """
        try:
            # Check for page number extraction
            if 'from page' in query.lower():
                page_num = int(query.lower().split('page')[1].split()[0])
                context = self.retrieve_page_content(page_num)
            else:
                # Default to full document retrieval
                context = ' '.join([doc['content'] for doc in self.documents])

            # Generate response
            prompt = f"""
            Context: {context}

            Query: {query}

            Provide a precise and factual response based strictly on the given context.
            If the information is not found, clearly state that.
            """

            response = self.llm.generate_content(prompt)
            return response.text

        except Exception as e:
            return f"Error processing query: {str(e)}"

def main():
    # Upload PDF in Colab using file upload widget
    from google.colab import files

    print("Please upload your PDF file")
    uploaded = files.upload()

    # Get the filename of the uploaded file
    pdf_filename = list(uploaded.keys())[0]

    # Initialize RAG Pipeline
    rag_pipeline = RAGPipeline()

    # Ingest PDF
    pdf_documents = rag_pipeline.extract_pdf_text(pdf_filename)
    rag_pipeline.index_documents(pdf_documents)

    # Print total number of pages for reference
    print(f"\nTotal Pages in PDF: {len(pdf_documents)}")

    # Interactive Query Loop
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")

        if query.lower() == 'exit':
            break

        response = rag_pipeline.generate_response(query)
        print("\nResponse:", response)

# Run the main function
if __name__ == "__main__":
    main()
