import os
import fitz
from openai import AzureOpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OAI_KEY"),
    api_version="2024-06-01",
    azure_endpoint=os.getenv("AZURE_OAI_ENDPOINT")
)

text_chunks = []
index = None

#---------------------- Extract text from PDF Function ----------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text("text")
    return text

#--------------------Chunk text into smaller pieces Function --------------------------
def chunk_text(text, max_chunk_size=2000):
    chunks = []
    words = text.split()
    current_chunk = []
    for word in words:
        if len(" ".join(current_chunk + [word])) > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ------------------------- Get embeddings for text Function -------------------------
def get_embeddings(text):
    response = client.embeddings.create(
        input=text,
        model="embed_chat"
    )
    embeddings = np.array(response.data[0].embedding)
    return embeddings

#-------------------------------- Create FAISS index function -------------------------
def create_faiss_index(embeddings, index_file="faiss_index.index"):
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_file)

#---------------------------- Perform similarity search function -------------------------
def similarity_search(query, index, text_chunks):
    query_embedding = get_embeddings(query)
    distances, indices = index.search(query_embedding.reshape(1, -1), k=2)
    results = [text_chunks[i] for i in indices[0]]
    return results

#-------------------------- Generate answer using GPT function -------------------------
def generate_answer(query, context):
    response = client.chat.completions.create(
        model="gpt-4o-2",
        messages=[
            {"role": "user", "content": f"Using the following context, answer the question: '{query}'\n\nContext: {context}"}
        ],
        temperature=0.7,
        max_tokens=150
    )
    answer = response.choices[0].message.content
    return answer

# ----------------------------- Process uploaded PDFs and create FAISS index--------------------------------
def setup_pdf_index(pdf_paths, index_file="faiss_index.index"):
    global text_chunks, index
    for pdf_path in pdf_paths:
        extracted_text = extract_text_from_pdf(pdf_path)
        pdf_text_chunks = chunk_text(extracted_text)
        embeddings = np.array([get_embeddings(chunk) for chunk in pdf_text_chunks])
        embeddings = np.vstack(embeddings)
        create_faiss_index(embeddings, index_file)
        text_chunks.extend(pdf_text_chunks)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global text_chunks, index
    files = request.files.getlist('files')  # The 'files' key matches the one in FormData

    if not files:
        return jsonify({'error': 'No files uploaded.'}), 400

    pdf_paths = []
    upload_dir = os.path.join(os.getcwd(), 'upload')  # Define your upload directory
    os.makedirs(upload_dir, exist_ok=True)  # Ensure the directory exists

    for file in files:
        pdf_path = os.path.join(upload_dir, file.filename)
        file.save(pdf_path)
        pdf_paths.append(pdf_path)

    setup_pdf_index(pdf_paths)  # Process the files for indexing
    index = faiss.read_index("faiss_index.index")  # Re-read the FAISS index

    return jsonify({'message': 'PDFs uploaded and processed successfully.'})

@app.route('/chat', methods=['POST'])
def chat_with_pdf():
    global text_chunks, index
    data = request.get_json()
    query = data.get('query', '')

    if not query or not text_chunks or index is None:
        return jsonify({'answer': 'Please upload a PDF first.'})

    results = similarity_search(query, index, text_chunks)
    context = "\n".join(results)
    answer = generate_answer(query, context)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
