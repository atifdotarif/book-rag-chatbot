import argparse
from pdf_loader import load_pdf
from chunker import chunk_docs
from vectorstore import build_or_load_vectorstore
from ragchain import build_rag_chain
from cli import chat_loop

def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot over a PDF book")
    parser.add_argument("--pdf", required=True, help="Path to the PDF book (>200 pages)")
    parser.add_argument("--index", default="book-chatbot", help="Pinecone index name")
    parser.add_argument("--top-k", type=int, default=4, help="Top K chunks to retrieve")
    args = parser.parse_args()

    docs, pages = load_pdf(args.pdf)
    print(f"[INFO] Loaded PDF with {pages} pages.")

    chunks = chunk_docs(docs)
    print(f"[INFO] Created {len(chunks)} chunks.")

    vectorstore = build_or_load_vectorstore(args.index, chunks)
    rag = build_rag_chain(vectorstore, top_k=args.top_k)

    chat_loop(rag)

if __name__ == "__main__":
    main()
