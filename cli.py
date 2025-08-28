def chat_loop(rag):
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
            print(f"\nAssistant: {rag(q)}\n")
    except KeyboardInterrupt:
        print("\nBye!")
