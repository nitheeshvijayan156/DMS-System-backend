from anthropic import Anthropic
from langchain_qdrant import Qdrant
from rag.embeddings import embeddings
from rag.qdrant_utils import client
import os

llm = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def query_llm(chat_name, query_text):
    vector_store = Qdrant(client=client, collection_name=chat_name, embeddings=embeddings)
    retriever = vector_store.as_retriever()
    docs = retriever.get_relevant_documents(query_text)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    full_prompt = (
        "You are a knowledgeable assistant. Your responses should only be based on "
        "the context below. If the query does not match the context, respond with "
        "'Your query does not match the context!'.\n\n"
        f"Context:\n{context}\n\n"
        f"User Query: {query_text}\n\n"
        "Answer:"
    )

    result = llm.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=512,
        temperature=0.3,
        messages=[{"role": "user", "content": full_prompt}]
    )

    return result.content[0].text.strip()

def classify_document_content(document_text):
    try:
        prompt = (
            "Classify this document into one of the following categories: "
            "Bill, Insurance, Invoice, Tax Document, Medical Report. "
            "Respond only with the exact category label, and do not add any punctuation or new categories. "
            f"\n\nDocument:\n{document_text}\n\nCategory:"
        )

        result = llm.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        return result.content[0].text.strip()
    except Exception as e:
        print(f"LLM classification error: {e}")
        return None
