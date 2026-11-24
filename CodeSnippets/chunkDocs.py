import os
import json
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
VECTOR_STORE_FILE = os.getenv("VECTOR_STORE_FILE")

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


# add the embeddings to the chunks of the documents
def addEmbeddings(chunks):
    for chunk in chunks:
        vector = getEmbeddings([chunk["text"]])
        chunk["vector"] = vector

    return chunks

# serch the documents for the most relevant document based on the query
def search(query, documents,top_k=3):
    query_vector = getEmbeddings([query])

    result = []
    for chunk in documents:
        score = cosineSimilarity(query_vector, chunk["vector"])
        result.append({
            "score": float(score),
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "size": len(chunk["text"])
        })

    result = sorted(result, key=lambda rec: rec["score"], reverse=True)
    return result[:top_k]



# take a one file and create chunks on the basic of section
def chunk_document(document):

    chunks = []
    chunk_id = 0
    for part in document["text"].split("\n\n"):
        part = part.strip()
        if part:
            chunks.append({
                "source": document["source"],
                "chunk_id": chunk_id,
                "text": part
            })
            chunk_id += 1

    return chunks

def create_chunks():
    documents = loadDocuments()
    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    return chunks


# testing method
def test(user_query,chunk,top_k= 3):
    print(f"Searching for: {user_query}")

    result = search(user_query, chunk, top_k)

    for r in result:
        print(f"{r['score']:.4f} - {r['source']} - {r['size']} char")

def save_vector_store():
    with open(VECTOR_STORE_FILE, 'w',encoding="UTF-8") as f:
        json.dump(chunks, f,indent=4)

def load_vector_store():
    with open(VECTOR_STORE_FILE, 'r',encoding="UTF-8") as f:
        return json.load(f) 

# test the chunking and embedding
# chunks = create_chunks()

if os.path.exists(VECTOR_STORE_FILE):
    chunks = load_vector_store()
    print(f"Loaded {len(chunks)} chunks from {VECTOR_STORE_FILE}") 
else:
    chunks = create_chunks()
    chunks = addEmbeddings(chunks)
    print(f"Created {len(chunks)} chunks from {KB_FOLDER} folder")
    save_vector_store()
 






print(f"We have embedded {len(chunks)} chunks from {KB_FOLDER} folder")

user_query = "What are the benefits of maternity leave?"

test(user_query, chunks,top_k= 3)