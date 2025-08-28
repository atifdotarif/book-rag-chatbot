# Book RAG Chatbot

A powerful RAG (Retrieval-Augmented Generation) chatbot that allows you to chat with PDF books using AI. Built with LangChain, OpenAI, and Pinecone.

## Features

- **PDF Processing**: Load and process large PDF books (200+ pages recommended)
- **Smart Chunking**: Intelligent text splitting with overlap for better context
- **Vector Search**: Semantic search using OpenAI embeddings
- **RAG Pipeline**: Grounded responses using retrieved context
- **Multiple Interfaces**: Both CLI and web UI available
- **Session Management**: Web UI with real-time progress tracking

## Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd book-rag-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

## Usage

### Web UI (Recommended)

1. Start the web application:
```bash
python run_web.py
```

2. Open your browser and go to `http://localhost:5000`

3. Upload a PDF book and wait for processing to complete

4. Start chatting with your book!

### CLI Version

1. Run the chatbot with a PDF:
```bash
python main.py --pdf your_book.pdf --index book-chatbot --top-k 4
```

2. Or use the monolithic version:
```bash
python rag_book_chat.py --pdf your_book.pdf --index book-chatbot --top-k 4
```

## Command Line Options

- `--pdf`: Path to the PDF file (required)
- `--index`: Pinecone index name (default: "book-chatbot")
- `--top-k`: Number of chunks to retrieve (default: 4)

## How It Works

1. **PDF Loading**: Uses PyPDFLoader to extract text from PDF pages
2. **Text Chunking**: Splits text into 1000-character chunks with 200-character overlap
3. **Embedding**: Converts chunks to vectors using OpenAI's text-embedding-3-small
4. **Vector Storage**: Stores embeddings in Pinecone vector database
5. **Retrieval**: Performs semantic search to find relevant chunks
6. **Generation**: Uses GPT-4o-mini to generate grounded responses

## Architecture

- **`app.py`**: Flask web application with session management
- **`main.py`**: CLI entry point with modular architecture
- **`rag_book_chat.py`**: Self-contained CLI version
- **`pdf_loader.py`**: PDF loading and validation
- **`chunker.py`**: Text chunking utilities
- **`vectorstore.py`**: Pinecone vector database operations
- **`ragchain.py`**: RAG pipeline construction
- **`config.py`**: Environment variable management
- **`cli.py`**: Command-line interface utilities

## Web UI Features

- **Drag & Drop**: Easy PDF upload with drag-and-drop support
- **Real-time Progress**: Live progress tracking during processing
- **Responsive Design**: Works on desktop and mobile devices
- **Session Management**: Each upload creates a unique session
- **Modern UI**: Beautiful, intuitive interface with smooth animations

## API Endpoints

- `GET /`: Main web interface
- `POST /upload`: Upload and process PDF
- `GET /status/<session_id>`: Get processing status
- `POST /chat`: Send chat message
- `POST /cleanup/<session_id>`: Clean up session

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key for embeddings and chat
- `PINECONE_API_KEY`: Your Pinecone API key for vector storage

## Dependencies

- **LangChain**: RAG pipeline framework
- **OpenAI**: Embeddings and chat models
- **Pinecone**: Vector database
- **Flask**: Web framework
- **PyPDF2**: PDF processing

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure both OpenAI and Pinecone API keys are set in `.env`
2. **PDF Too Small**: The system works best with books of 200+ pages
3. **Processing Errors**: Check that your PDF is not corrupted and is text-based
4. **Memory Issues**: Large PDFs may require more RAM

### Error Messages

- "I could not find it": The answer is not present in the retrieved context
- "Session not found": The web session has expired or been cleaned up
- "RAG chain not ready": PDF processing is still in progress

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please open an issue on the repository.
