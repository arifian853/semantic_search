from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

documents = [
    "Python adalah bahasa pemrograman.", 
    "Machine Learning memungkinkan komputer belajar dari data.", 
    "FastAPI digunakan untuk membuat backend API dengan Python.", 
    "Kucing adalah hewan mamalia.",
    "Neural Network merupakan salah satu metode Deep Learning."
]

document_embeddings = model.encode(documents)

print(document_embeddings.shape)

query = "Hewan yang termasuk mamalia"

query_embedding = model.encode(query)

scores = model.similarity(
    query_embedding,
    document_embeddings
)

print(scores)

best_index = scores.argmax().item()

best_document = documents[best_index]

best_score = scores[0][best_index].item()

print("\nQuery:")
print(query)
print("\nHasil paling relevan:")
print(best_document)
print("\nSimilarity Score:")
print(best_score)