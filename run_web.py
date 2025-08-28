"""
Web UI startup script for Book RAG Chatbot
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check required environment variables
required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    print("Missing required environment variables:")
    for var in missing_vars:
        print(f"   - {var}")
    print("\nPlease create a .env file with your API keys:")
    print("OPENAI_API_KEY=your_openai_key_here")
    print("PINECONE_API_KEY=your_pinecone_key_here")
    sys.exit(1)

print("✅ Environment variables loaded successfully!")
print("🚀 Starting Book RAG Chatbot Web UI...")
print("📱 Open your browser and go to: http://localhost:5000")
print("⏹️  Press Ctrl+C to stop the server\n")

# Import and run the Flask app
from app import app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
