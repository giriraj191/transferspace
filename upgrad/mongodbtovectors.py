#    pip install pymongo sentence-transformers scikit-learn numpy

import pymongo
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["vectordb_demo"]
collection = db["hero_embeddings"]
model = SentenceTransformer("bert-base-nli-mean-tokens")
def populate_data():
    data = [
        {"text": "Tony Stark builds the Iron Man suit", "category": "technology"},
        {"text": "Steve Rogers leads the Avengers against Thanos", "category": "leadership"},
        {"text": "Thor wields Mjolnir and protects Asgard", "category": "mythology"},
        {"text": "Bruce Banner transforms into the Hulk", "category": "science"},
        {"text": "Natasha Romanoff is a skilled spy and assassin", "category": "espionage"},
    ]
    for entry in data:
        entry["embedding"] = model.encode(entry["text"]).tolist()  # Generate embeddings
        collection.insert_one(entry)
    print("Avengers Universe data inserted into MongoDB.")

# populate_data()
# print(f"Data is inserted into MongoDB")
def similarity_search(query_text, top_n=3):
    query_embeddings = model.encode(query_text).reshape(1,-1)
    documents = list(collection.find())
    similarities = []
    for doc in documents:
        doc_embedding = np.array(doc["embedding"]).reshape(1,-1)
        similarity = cosine_similarity(query_embeddings, doc_embedding)[0][0]
        similarities.append({ "text":doc["text"], "category": doc["category"], "similarity": similarity })
        similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_n]

query = "A genius inventor creates advanced technology"
print(f"Query: {query}")
similarity_entries = similarity_search(query, top_n=3)

print(f"Similarity Result")
for entry in similarity_entries:
    print(f"Text: {entry['text']}, Category: {entry['category']}, Similarity: {entry['similarity']}")


query = "A sinful killer and a spy"
print(f"Query: {query}")
similarity_entries_thor = similarity_search(query, top_n=3)

print(f"Similarity Result")
for entry in similarity_entries_thor:
    print(f"Text: {entry['text']}, Category: {entry['category']}, Similarity: {entry['similarity']}")