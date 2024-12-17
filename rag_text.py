
import os
import requests
from bs4 import BeautifulSoup
import uuid
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify

# Initialize the Sentence Transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
dimension = 384  # Embedding size of MiniLM
index = faiss.IndexFlatL2(dimension)

# Metadata store
meta_store = []

# Function to scrape and chunk content from a website
def scrape_and_chunk(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract textual content
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    content = " ".join(paragraphs)

    # Split content into chunks (e.g., 300 words per chunk)
    chunk_size = 300
    words = content.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    # Generate embeddings for each chunk
    embeddings = [embedding_model.encode(chunk) for chunk in chunks]

    # Create metadata for each chunk
    metadata = [{"id": str(uuid.uuid4()), "url": url, "chunk": chunk} for chunk in chunks]

    return embeddings, metadata

# Function to add data to the FAISS index
def add_to_index(embeddings, metadata, index, meta_store):
    if embeddings:
        vectors = np.array(embeddings).astype('float32')
        index.add(vectors)

        # Store metadata separately
        meta_store.extend(metadata)

# Function to perform similarity search
def search(query, index, meta_store, top_k=5):
    query_embedding = embedding_model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve metadata for the top results
    results = [meta_store[i] for i in indices[0]]
    return results

# Flask app setup
app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data['query']

    # Perform search and generate response
    results = search(query, index, meta_store)
    response_content = "\n".join([f"- {chunk['chunk']}" for chunk in results])

    return jsonify({"response": response_content})

if __name__ == '__main__':
    # Start Flask app in the background
    from threading import Thread
    def run_flask():
        app.run(debug=True, use_reloader=False)  # Set use_reloader=False to avoid double startup
    thread = Thread(target=run_flask)
    thread.start()

    # Prompt for user input dynamically
    while True:
        url = input("Enter the website URL to scrape: ")
        query = input("Enter your query: ")

        # Scrape and index the website
        embeddings, metadata = scrape_and_chunk(url)
        add_to_index(embeddings, metadata, index, meta_store)

        # Send the query to the Flask server
        response = requests.post("http://127.0.0.1:5000/query", json={"query": query})
        if response.status_code == 200:
            print("Response:", response.json()["response"])
        else:
            print("Error:", response.status_code)

        # Optionally, you can break the loop by typing 'exit'
        if query.lower() == 'exit':
            print("Exiting...")
            break
