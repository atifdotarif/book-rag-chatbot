import os, sys
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(pdf_path: str):
    if not os.path.isabs(pdf_path):
        pdf_path = os.path.join(os.getcwd(), pdf_path)
    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)

    docs = PyPDFLoader(pdf_path).load()
    pages = len(docs)
    if pages < 200:
        print(f"[WARNING] Book has only {pages} pages (<200).")
    return docs, pages
