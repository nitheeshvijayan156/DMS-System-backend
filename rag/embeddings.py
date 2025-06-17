from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from rag.qdrant_utils import create_qdrant_collection, collection_exists, client

model_name="sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=model_name, 
                                model_kwargs={'device': 'cpu'},show_progress=True)

# Function to store embeddings and load old ones for a chat
def handle_chat_embeddings(chat_name, document_text=None):

    # Check if the collection exists
    if collection_exists(chat_name):
        print(f"Collection '{chat_name}' exists. Loading old embeddings.")
        vector_store = Qdrant(client=client, collection_name=chat_name, embeddings=embeddings)
    else:
        print(f"Creating new collection for chat '{chat_name}'")
        create_qdrant_collection(chat_name)
        vector_store = Qdrant(client=client, collection_name=chat_name, embeddings=embeddings)
    
    # If a document is uploaded, create new embeddings and add to the collection
    if document_text:
        text_splitter=CharacterTextSplitter(
                        separator="\n",
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                        )
        chunks = text_splitter.split_text(document_text)

        for chunk in chunks:
            vector_store.add_texts([chunk])
        
        print(f"New document embeddings stored in collection '{chat_name}'")

    return vector_store  