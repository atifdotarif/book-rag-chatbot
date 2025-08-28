from flask import Flask, render_template, request, jsonify, session
import os
import tempfile
from werkzeug.utils import secure_filename
import threading
import time
from pdf_loader import load_pdf
from chunker import chunk_docs
from vectorstore import build_or_load_vectorstore
from ragchain import build_rag_chain
import uuid

app = Flask(__name__)
app.secret_key = 'secret-key'  # for production

# Global storage for active sessions
active_sessions = {}

class ProcessingSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.status = "initialized"
        self.progress = 0
        self.message = ""
        self.rag_chain = None
        self.error = None
        
    def update_status(self, status, progress, message):
        self.status = status
        self.progress = progress
        self.message = message

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Please upload a PDF file'}), 400
    
    # Create unique session ID
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    
    # Create processing session
    active_sessions[session_id] = ProcessingSession(session_id)
    
    # Save file temporarily
    filename = secure_filename(file.filename)
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)
    file.save(temp_path)
    
    # Start processing in background thread
    thread = threading.Thread(target=process_pdf, args=(temp_path, session_id))
    thread.daemon = True
    thread.start()
    
    return jsonify({'session_id': session_id, 'status': 'processing_started'})

def process_pdf(pdf_path, session_id):
    """Process PDF in background thread"""
    session_data = active_sessions[session_id]
    
    try:
        # Step 1: Load PDF
        session_data.update_status("loading", 10, "Loading PDF file...")
        docs, pages = load_pdf(pdf_path)
        session_data.update_status("loading", 20, f"Loaded PDF with {pages} pages")
        
        # Step 2: Chunk documents
        session_data.update_status("chunking", 40, "Creating text chunks...")
        chunks = chunk_docs(docs)
        session_data.update_status("chunking", 60, f"Created {len(chunks)} chunks")
        
        # Step 3: Build vector store
        session_data.update_status("embedding", 80, "Building vector store...")
        vectorstore = build_or_load_vectorstore(f"web-{session_id}", chunks)
        session_data.update_status("embedding", 90, "Vector store ready")
        
        # Step 4: Build RAG chain
        session_data.update_status("finalizing", 95, "Building RAG chain...")
        rag_chain = build_rag_chain(vectorstore, top_k=4)
        session_data.rag_chain = rag_chain
        
        # Step 5: Complete
        session_data.update_status("ready", 100, "Ready to chat!")
        
    except Exception as e:
        session_data.error = str(e)
        session_data.update_status("error", 0, f"Error: {str(e)}")
    finally:
        # Clean up temp file
        try:
            os.remove(pdf_path)
            os.rmdir(os.path.dirname(pdf_path))
        except:
            pass

@app.route('/status/<session_id>')
def get_status(session_id):
    if session_id not in active_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session_data = active_sessions[session_id]
    return jsonify({
        'status': session_data.status,
        'progress': session_data.progress,
        'message': session_data.message,
        'error': session_data.error
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    session_id = data.get('session_id')
    question = data.get('question')
    
    if not session_id or not question:
        return jsonify({'error': 'Missing session_id or question'}), 400
    
    if session_id not in active_sessions:
        return jsonify({'error': 'Session not found'}), 404
    
    session_data = active_sessions[session_id]
    
    if session_data.status != 'ready':
        return jsonify({'error': 'RAG chain not ready yet'}), 400
    
    try:
        answer = session_data.rag_chain(question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': f'Chat error: {str(e)}'}), 500

@app.route('/cleanup/<session_id>', methods=['POST'])
def cleanup_session(session_id):
    if session_id in active_sessions:
        del active_sessions[session_id]
        return jsonify({'status': 'cleaned'})
    return jsonify({'error': 'Session not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
