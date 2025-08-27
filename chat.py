def run_chat(rag):
    print("\n📘 Book RAG Chatbot")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break
        print(f"\nAssistant: {rag(q)}\n")
