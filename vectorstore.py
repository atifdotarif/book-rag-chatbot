from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from config import PINECONE_API_KEY

def ensure_index(pc: Pinecone, name: str, dimension: int):
    existing = {idx["name"] for idx in pc.list_indexes()}
    if name not in existing:
        print(f"[INFO] Creating Pinecone index '{name}' (dim={dimension})")
        pc.create_index(name=name, dimension=dimension, metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    else:
        print(f"[INFO] Using existing Pinecone index '{name}'.")

def build_vectorstore(index_name: str, chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pc, index_name, dimension=1536)
    return PineconeVectorStore.from_documents(chunks, embeddings, index_name)
