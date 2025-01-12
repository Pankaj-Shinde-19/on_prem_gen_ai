from flask import Flask, request, jsonify
import time  # Import time module to measure execution time
import pickle
import requests
import json
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from collections import deque  # Import deque for maintaining conversation history

# Disable symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize Flask app
app = Flask(__name__)

# Define file paths for loading preprocessed data
embedding_files_path = r"C:\Users\PankajShinde\PycharmProjects\testing-on-premise-gen-ai\embedding_files\file_name"
corpus_file_path = os.path.join(embedding_files_path, "corpus.txt")
chunks_file_path = os.path.join(embedding_files_path, "chunks.pkl")
embeddings_file_path = os.path.join(embedding_files_path, "embeddings.pkl")

# Load preprocessed data
with open(corpus_file_path, "r") as f:
    corpus = f.read()

with open(chunks_file_path, "rb") as f:
    chunks = pickle.load(f)

# Load models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device='cpu')

# Initialize Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)

# Maintain a conversation history (deque for efficient appends and pops)
conversation_history = deque(maxlen=2)

# Retrieval function
def retrieve_documents(query, top_k=5):
    query_vector = embedding_model.encode([query])[0]
    results = qdrant_client.search(
        collection_name="documents",
        query_vector=query_vector,
        limit=top_k
    )
    candidate_docs = [result.payload["text"] for result in results]
    if candidate_docs:
        pairs = [[query, doc] for doc in candidate_docs]
        scores = cross_encoder.predict(pairs)
        ranked_results = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked_results]
    return []

# Response generation function using phi3:mini
def generate_response(context, query, history):
    url = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json"
    }
    # Format conversation history as part of the prompt
    history_text = "\n".join([f"Q: {h['query']}\nA: {h['response']}" for h in history])
    prompt = f"Conversation History:\n{history_text}\nContext: {context} Question: {query} Provide a brief and direct answer:"
    payload = {
        "model": "phi3:mini",
        "prompt": prompt
    }

    try:
        # Make the POST request
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        raw_data = response.text.strip().split("\n")
        output = []
        for item in raw_data:
            try:
                data = json.loads(item)
                if 'response' in data:
                    output.append(data['response'])
            except json.JSONDecodeError:
                print(f"Failed to parse: {item}")
        return "".join(output)  # Combine the extracted responses
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while generating the response."

# Flask route for API
@app.route("/ask", methods=["POST"])
def ask():
    """
    Handles user queries and measures response time for retrieval and response generation.

    Returns:
        JSON response containing the generated answer, timing breakdowns, and time taken.
    """
    data = request.get_json()
    query = data.get("query", "")

    if query:
        start_time = time.time()  # Start overall timing

        # Step 1: Retrieve documents
        start_retrieval = time.time()
        retrieved_docs = retrieve_documents(query)
        retrieval_time = time.time() - start_retrieval

        # Step 2: Generate response
        context = " ".join(retrieved_docs)  # Combine context
        start_generation = time.time()
        response = generate_response(context, query, conversation_history)
        generation_time = time.time() - start_generation

        # Step 3: Update conversation history
        conversation_history.append({"query": query, "response": response})

        end_time = time.time()  # End overall timing
        total_time = round(end_time - start_time, 2)

        return jsonify({
            "response": response,
            "retrieval_time": f"{retrieval_time:.2f} seconds",
            "generation_time": f"{generation_time:.2f} seconds",
            "total_time": f"{total_time:.2f} seconds"
        })
    else:
        return jsonify({"error": "No query provided"}), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
