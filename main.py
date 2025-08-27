import argparse
from loaders import load_pdf
from chunker import chunk_docs
from vectorstore import build_vectorstore
from rag_chain import build_rag_chain
from chat import run_chat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--index", default="book-chatbot")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    docs, pages = load_pdf(args.pdf)
    print(f"[INFO] Loaded PDF with {pages} pages.")
    chunks = chunk_docs(docs)
    print(f"[INFO] Created {len(chunks)} chunks.")

    vs = build_vectorstore(args.index, chunks)
    rag = build_rag_chain(vs, top_k=args.top_k)
    run_chat(rag)

if __name__ == "__main__":
    main()
