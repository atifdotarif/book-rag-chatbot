from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def build_rag_chain(vectorstore, top_k=4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful Q&A assistant for a book.
Use only the context provided. If not present, reply:
"I could not find it"

Context:
{context}

Question:
{question}

Answer:"""
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def format_docs(docs: List) -> str:
        return "\n\n---\n\n".join(
            f"[Page {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
            for d in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    def guarded_invoke(question: str):
        docs = retriever.invoke(question)
        if not docs:
            return "I could not find it"
        return chain.invoke(question)

    return guarded_invoke
