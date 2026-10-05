from sentence_transformers import SentenceTransformer
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

Melihat bentuk embedding dari sebuah teks
text = "Saya sedang belajar Embedding"
embedding = model.encode(text)
print(embedding)
print(embedding.shape)

text1 = "Saya suka bermain sepak bola"  # Suka main bola
text2 = "Saya sedang belajar Python"      # Senang main bola 
embedding1 = model.encode(text1)
embedding2 = model.encode(text2)

similarity = model.similarity(
    embedding1,
    embedding2
)

print("Similarity of text1 and text2:", similarity)

# Compute cosine similarity and Euclidean distance
import numpy as np

cosine = np.dot(embedding1, embedding2) / (
    np.linalg.norm(embedding1)
    * np.linalg.norm(embedding2)
)

distance = np.linalg.norm(
    embedding1 - embedding2
)

print("Cosine Similarity:", cosine)
print("Euclidean Distance:", distance)