from flask import Flask, request, jsonify
import time
import pickle
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import BartForConditionalGeneration, BartTokenizer
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
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

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


# Response generation function
def generate_response(context, query, history):
    # Prepare conversation history as part of the context
    history_text = "\n".join([f"Q: {h['query']}\nA: {h['response']}" for h in history])
    input_text = f"Conversation History:\n{history_text}\nContext: {context} Question: {query} Please provide a brief and clear answer, ensuring the response directly addresses the question."

    # Tokenize and generate response using BART
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(
        inputs.input_ids, max_length=150, min_length=30,
        length_penalty=2.0, num_beams=6, early_stopping=True
    )
    refined_output = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
    return refined_output


# Flask route for API
@app.route("/ask", methods=["POST"])
def ask():
    """
    Handles user queries and measures response time.

    Returns:
        JSON response containing the generated answer, timing breakdowns, and time taken.
    """
    data = request.get_json()
    query = data.get("query", "")

    if query:
        start_time = time.time()  # Start timing

        # Step 1: Retrieve documents
        retrieved_docs = retrieve_documents(query)
        context = " ".join(retrieved_docs)  # Combine context

        # Step 2: Generate response
        response = generate_response(context, query, conversation_history)

        # Step 3: Add current question and response to conversation history
        conversation_history.append({"query": query, "response": response})

        end_time = time.time()  # End timing
        time_taken = round(end_time - start_time, 2)  # Calculate elapsed time

        return jsonify({"response": response, "time_taken": f"{time_taken} seconds"})
    else:
        return jsonify({"error": "No query provided"}), 400


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
