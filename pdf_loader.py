import os, sys
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path: str):
    if not os.path.isabs(pdf_path):
        pdf_path = os.path.join(os.getcwd(), pdf_path)

    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF not found at: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    num_pages = len(docs)

    if num_pages < 200:
        print(f"[WARNING] The PDF has only {num_pages} pages (<200).", file=sys.stderr)

    return docs, num_pages
