# rag_book_chat.py
# ---------------------------------------------------------
# RAG Chatbot over a (>=200 pages) PDF book using:
# LangChain + OpenAI (embeddings & chat) + Pinecone (vector DB)
#
# Features:
# - Loads & validates big PDF
# - Chunks with overlap
# - Embeds to Pinecone (creates index if missing)
# - Semantic search + strict grounded generation
# - CLI chat loop; type 'exit' to quit
#
# Env vars required:
#   OPENAI_API_KEY, PINECONE_API_KEY
# Optional flags:
#   --pdf /path/to/book.pdf
#   --index book-chatbot
#   --top-k 4
# ---------------------------------------------------------

import os
import argparse
import sys
from typing import List

# ---- LangChain core pieces ----
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---- Pinecone v5 SDK ----
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()  # will load variables from .env file in same dir


# ---------------------------
# Utility: fail-fast checks
# ---------------------------
def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        print(f"[ERROR] Environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


# ---------------------------
# 1) Load & validate PDF
# ---------------------------
def load_pdf(pdf_path: str):
    # If only filename given, assume current directory
    if not os.path.isabs(pdf_path):
        pdf_path = os.path.join(os.getcwd(), pdf_path)

    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF not found at: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    num_pages = len(docs)
    if num_pages < 200:
        print(
            f"[WARNING] The PDF has only {num_pages} pages (< 200). "
            f"Your internship task requires a book > 200 pages.",
            file=sys.stderr,
        )
    return docs, num_pages


# ---------------------------
# 2) Chunk documents
# ---------------------------
def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


# ---------------------------
# 3) Prepare embeddings + Pinecone index
# ---------------------------
def ensure_pinecone_index(pc: Pinecone, index_name: str, dimension: int):
    existing = {idx["name"] for idx in pc.list_indexes()}
    if index_name not in existing:
        print(f"[INFO] Creating Pinecone index '{index_name}' (dim={dimension}) ...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # NOTE: Small indexes initialize quickly; for larger ones you might need to wait.
    else:
        print(f"[INFO] Using existing Pinecone index '{index_name}'.")


# ---------------------------
# 4) Build / load vector store
# ---------------------------
def build_or_load_vectorstore(index_name: str, chunks):
    # Choose embedding model; dim=1536 for text-embedding-3-small
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Ensure Pinecone index exists with correct dimension
    pc = Pinecone(api_key=require_env("PINECONE_API_KEY"))
    ensure_pinecone_index(pc, index_name, dimension=1536)

    # Create / connect vector store
    print("[INFO] Ingesting chunks into Pinecone (if not already present) ...")
    vs = PineconeVectorStore.from_documents(
        documents=chunks, embedding=embeddings, index_name=index_name
    )
    return vs


# ---------------------------
# 5) Build RAG chain (strict)
# ---------------------------
def build_rag_chain(vectorstore, top_k: int = 4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # Prompt that strictly forbids answering outside the given context
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful Q&A assistant for a book.
You must answer **only** using the provided context (excerpts from the book).
If the answer is not present in the context, respond exactly with:
"I could not find it"

Guidelines:
- Do not use any outside knowledge.
- If context is partial/insufficient, still respond exactly with: "I could not find it".
- Keep answers concise and directly cite details only from context.

Context:
{context}

Question:
{question}

Answer:"""
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def format_docs(docs) -> str:
        # Join page content and optional metadata for transparency
        parts: List[str] = []
        for d in docs:
            page = d.metadata.get("page", "N/A")
            parts.append(f"[Page {page}]\n{d.page_content}".strip())
        return "\n\n---\n\n".join(parts)

    # LCEL pipeline
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrap with a guard: if retriever returns nothing, short-circuit.
    def guarded_invoke(question: str):
        docs = retriever.invoke(question)
        if not docs:
            return "I could not find it"
        # else run full chain (we already fetched docs once; but retriever will run again in chain).
        # To avoid double search, we could inline, but keeping simple here:
        return chain.invoke(question)

    return guarded_invoke


# ---------------------------
# 6) Main CLI loop
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot over a PDF book")
    parser.add_argument("--pdf", required=True, help="Path to the PDF book (>200 pages)")
    parser.add_argument("--index", default="book-chatbot", help="Pinecone index name")
    parser.add_argument("--top-k", type=int, default=4, help="Top K chunks to retrieve")
    args = parser.parse_args()

    # Check env vars
    require_env("OPENAI_API_KEY")
    require_env("PINECONE_API_KEY")

    # Load -> chunk -> embed/store
    docs, pages = load_pdf(args.pdf)
    print(f"[INFO] Loaded PDF with {pages} pages.")
    chunks = chunk_docs(docs)
    print(f"[INFO] Created {len(chunks)} chunks.")

    vectorstore = build_or_load_vectorstore(args.index, chunks)
    rag = build_rag_chain(vectorstore, top_k=args.top_k)

    print("\n📘 Book RAG Chatbot")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    try:
        while True:
            q = input("You: ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit", "q"}:
                print("Bye!")
                break
            answer = rag(q)
            print(f"\nAssistant: {answer}\n")
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()
