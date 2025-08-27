from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)
