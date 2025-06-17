from dotenv import load_dotenv
from embeddings import handle_chat_embeddings
from llm import query_llm

load_dotenv()

def extract_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        return file.read()

if __name__ == "__main__":
    chat_name = input("Enter chat name: ") 

    upload_document = input("Upload a new document? (y/n): ").strip().lower()
    document_text = None
    if upload_document == "y":
        file_path = input("Enter the document file path: ")
        document_text = extract_text_from_file(file_path)  

    vector_store = handle_chat_embeddings(chat_name, document_text)

    query_text = input("Enter your query: ")  
    response = query_llm(chat_name, query_text)
    
    print(f"LLM Response: {response['result']}")
