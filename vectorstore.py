from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from config import PINECONE_API_KEY

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
    else:
        print(f"[INFO] Using existing Pinecone index '{index_name}'.")

def build_or_load_vectorstore(index_name: str, chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_pinecone_index(pc, index_name, dimension=1536)

    print("[INFO] Ingesting chunks into Pinecone (if not already present) ...")
    return PineconeVectorStore.from_documents(
        documents=chunks, embedding=embeddings, index_name=index_name
    )
