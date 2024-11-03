from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model from HuggingFace Hub
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(texts):
    if isinstance(texts, str):
        texts = [texts]
    
    # Returns numpy array if convert_to_tensor=False
    text_embeddings = model.encode(texts, convert_to_tensor=False)  
    return np.array(text_embeddings)


# Vector Storage
def vectorStorage(embeddings):
    # Check if embeddings is a 2D numpy array
    assert len(embeddings.shape) == 2, "Embeddings should be a 2D array"
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  
    index.add(embeddings)  
    return index


# def search_index(index, query, texts, top_k=1):
#     query_embedding = embed_text(query)

#     distances, indices = index.search(query_embedding, top_k)

#     closest_texts = [texts[i] for i in indices[0]]

#     return closest_texts, distances


# texts = ["What is LangGraph?", "LangGraph is an agent framework for AI project management."]
# embeddings = embed_text(texts)
# index = vectorStorage(embeddings)

# query = "What can LangGraph do?"
# closest_texts, distances = search_index(index, query, texts)

# print("Closest texts:")
# for text, distance in zip(closest_texts, distances[0]):
#     print(f"Text: {text}, Distance: {distance}")
