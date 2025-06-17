import os
from dotenv import load_dotenv
from anthropic import Anthropic

# Load environment variables
load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

VALID_CATEGORIES = [
    "Medical", "Insurance", "Finance", "Utility", "Legal", "Hotel", "Retail", "Others"
]

def clean_and_validate_response(response_text: str) -> str:
    if not response_text:
        return "Others"
    cleaned = response_text.strip().split(".")[0].strip()
    for category in VALID_CATEGORIES:
        if cleaned.lower() == category.lower():
            return category
    return "Others"

def classify_document_content(document_text: str) -> str:
    prompt = (
        "You are a document classifier. Classify the following document into one of these exact categories: "
        "Medical, Insurance, Finance, Utility, Legal, Hotel, Retail, Others. "
        "Respond ONLY with the category name. No extra text, punctuation, or explanation.\n\n"
        f"{document_text}\n\n"
        "Category:"
    )

    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=10,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )

    return clean_and_validate_response(response.content[0].text)