import qdrant_client
from qdrant_client.http import models
from dotenv import load_dotenv
import os

load_dotenv()
# Initialize Qdrant client
client = qdrant_client.QdrantClient(
    os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Function to create a new collection in Qdrant using chat name
def create_qdrant_collection(chat_name):
    vectors_config = models.VectorParams(size=384, distance=models.Distance.COSINE)
    client.create_collection(
        collection_name=chat_name,
        vectors_config=vectors_config,
    )
    print(f"Collection '{chat_name}' created successfully.")

# Function to check if a collection exists
def collection_exists(chat_name):
    collections = client.get_collections()
    for collection in collections.collections:
        if collection.name == chat_name:
            return True
    return False