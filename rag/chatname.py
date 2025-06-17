import requests
from langchain_core.runnables import RunnableLambda
import os
from dotenv import load_dotenv
from anthropic import Anthropic
load_dotenv()

# class CustomLLMBatch(RunnableLambda):
#     def __init__(self, api_url: str, api_key: str):
#         self.api_url = api_url
#         self.api_key = api_key

#     def invoke(self, query: str):

#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }
#         payload = {
#             "query": query
#         }

#         response = requests.post(self.api_url, json=payload, headers=headers)
#         if response.status_code == 200:
#             data = response.json()
#             return data  
#         else:
#             print("Response Body:", response.text)  
#             return f"Error: Unable to retrieve response, status code: {response.status_code}"

#     def __call__(self, query: str):
#         return self.invoke(query)
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def create_chat_name(document_content: str, user_query: str) -> str:
    prompt = (
        "Based on the document content and user query below, generate a concise cool chat name that is 1-3 words long. "
        "Please do not include any explanations, alternatives, or additional responses. Just provide the chat name.\n\n"
        f"Document Content: {document_content}\n"
        f"User Query: {user_query}\n\n"
        "Chat Name:"
    )

    result = client.messages.create(
        model="claude-3-opus-20240229",  # replace with your actual model
        max_tokens=10,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )

    print("LLM Response:", result)

    try:
        chat_name = result.content[0].text.strip()
        return chat_name
    except Exception as e:
        print("Error in response:", result)
        return "Error: Unable to generate chat name."


# # Example usage
# if __name__ == "__main__":
# #     api_url = os.getenv("API_BATCH_URL")
# #     api_key = os.getenv("API_KEY")
    
# #     llm = llm(api_url=api_url, api_key=api_key)

# #     document_content = '''Once upon a time, in a small town, lived a brave and curious little boy named Jack. Jack was always up for an adventure and loved exploring new places. One day, while playing in his backyard, he discovered an old map hidden in a dusty box. The map showed a mysterious island called Treasure Cove.

# # Excited by the prospect of treasure, Jack decided to embark on a thrilling expedition to find this hidden island. Armed with his backpack filled with snacks and his faithful magnifying glass, he set off on his journey.
# # Jack hopped on a small wooden boat with his best friend, Emma. Emma was a smart and resourceful girl who always had a plan. Together, they sailed across the vast blue ocean, marveling at the beauty of the sea and the creatures that swam beneath them.

# # After several days of sailing, they arrived at Treasure Cove. The island was lush with tropical plants, tall palm trees, and colorful flowers. The air was filled with the sweet scent of exotic fruits.
# # As they explored the island, Jack and Emma stumbled upon an old, crumbling temple. Inside, they found a clue that led them to believe there was a secret passage hidden somewhere on the island. Determined to find it, they followed a trail of ancient symbols etched into the rocks.

# # They encountered many challenges along the way. They had to solve puzzles, cross treacherous bridges, and even face a mischievous tribe of monkeys guarding the entrance to the secret passage. But Jack and Emma never gave up; they used their intelligence and bravery to overcome each obstacle.

# # Just as they were about to reach the secret passage, a fierce storm brewed on the horizon. The wind howled, and the waves crashed against the shore. Jack and Emma had to take shelter in a nearby cave, hoping the storm would pass.

# # Inside the cave, they found an old pirate's journal. It spoke of the hidden treasure that lay beyond the secret passage. The storm raged on for days, but Jack and Emma didn't lose hope. They studied the journal, learning about the island's history and the clues to finding the passage.
# # '''
    
# #     user_query = "What is the moral of this story?"

#     chat_name = create_chat_name(document_content, user_query, llm)
#     print(f"Generated Chat Name: {chat_name}") 