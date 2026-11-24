import os 
import numpy
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()


def cosineSimilarity(a,b):
    dot = numpy.dot(a,b)
    lenA = numpy.linalg.norm(a)
    lenB = numpy.linalg.norm(b)

    return dot/(lenA*lenB)

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)


def loadDocuments(folder):
    documents = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                documents.append({
                    "source": filename,
                    "text":f.read()
                })
    return documents

def addEmbeddings(documents):
    for doc in documents:
        vector = getEmbeddings([doc["text"]])
        doc["vector"] = vector

    return documents

def getEmbeddings(texts):
    response = client.embeddings.create(
        model=os.getenv("EMBED_MODEL"),
        input=texts
    )
    return response.data[0].embedding


# for i,embed in enumerate(response.data):
#     print(f"Embedding {i} of length: {len(embed.embedding)}")
#     print(embed.embedding[:5])


embeddedDocuments = addEmbeddings(loadDocuments(os.getenv("KB_FOLDER")))

print(embeddedDocuments.documents);


