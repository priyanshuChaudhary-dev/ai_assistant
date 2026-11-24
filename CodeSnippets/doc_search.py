import os
import numpy
from dotenv import load_dotenv

load_dotenv()
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
)

Embed_model = os.getenv("EMBED_MODEL")  
KB_FOLDER = os.getenv("KB_FOLDER")

# get the cosine similarity between two vectors
def cosineSimilarity(a,b):
    dot = numpy.dot(a,b)
    lenA = numpy.linalg.norm(a)
    lenB = numpy.linalg.norm(b)

    return dot/(lenA*lenB)


# get the embeddings for the documents
def getEmbeddings(texts):
    response = client.embeddings.create(
        model=Embed_model,
        input=texts
    )
    return response.data[0].embedding

# load the documents from the knowledge base folder
def loadDocuments(folder=KB_FOLDER):
    documents = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                documents.append({
                    "source": filename,
                    "text":f.read()
                })
    return documents    


# add the embeddings to the documents
def addEmbeddings(documents):
    for doc in documents:
        vector = getEmbeddings([doc["text"]])
        doc["vector"] = vector

    return documents

# serch the documents for the most relevant document based on the query
def searchDocuments(query, documents,top_k=3):
    query_vector = getEmbeddings([query])

    result = []
    for doc in documents:
        score = cosineSimilarity(query_vector, doc["vector"])
        result.append({
            "score": float(score),
            "source": doc["source"],
            "text": doc["text"],
            "size": len(doc["text"])
        })

    result = sorted(result, key=lambda rec: rec["score"], reverse=True)
    return result[:top_k]

# testing method
def test(user_query,documents,top_k= 3):
    print(f"Searching for: {user_query}")

    result = searchDocuments(user_query, documents, top_k)

    for r in result:
        print(f"{r['score']:.4f} - {r['source']} - {r['size']} char")

    


# let,s start the testing
documents = loadDocuments()
print(documents[0])

# documents = addEmbeddings(documents)

# print(f"Loaded {len(documents)} documents from {KB_FOLDER}")


# user_query = "What is the sick leave policy?"
# test(user_query, documents)