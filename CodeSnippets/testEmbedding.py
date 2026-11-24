import os 
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

response = client.embeddings.create(
    model=os.getenv("EMBED_MODEL"),
    input=[
        "I am a good coder",
        "Jaat da mukabala",
        "Jaat"
    ]
)

for i,embed in enumerate(response.data):
    print(f"Embedding {i} of length: {len(embed.embedding)}")
    print(embed.embedding[:5])

