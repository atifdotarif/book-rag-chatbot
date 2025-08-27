import os, sys
from dotenv import load_dotenv
load_dotenv()

def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        print(f"[ERROR] Env var {name} not set", file=sys.stderr)
        sys.exit(1)
    return val

OPENAI_API_KEY = require_env("OPENAI_API_KEY")
PINECONE_API_KEY = require_env("PINECONE_API_KEY")
