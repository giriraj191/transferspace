# python -m venv myvenv
# source myvenv/bin/activate
# On Windows - activate myvenv\Scripts\activate
# pip install numpy scikit-learn

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

vector_a = np.array([1, 2, 3])
vector_b = np.array([4, 5, 6])

dot_product =np.dot(vector_a, vector_b)
magnitude_a = np.linalg.norm(vector_a)
magnitude_b = np.linalg.norm(vector_b)
consine_similarity_manual = dot_product / (magnitude_a * magnitude_b)
print(f"Manual Cosine Similarity: {consine_similarity_manual}")
# reshaping the vectors
vector_a_reshaped = vector_a.reshape(1, -1)
vector_b_reshaped = vector_b.reshape(1, -1)
cosine_similarity_sklearn = cosine_similarity(vector_a_reshaped, vector_b_reshaped)
print(f"Scikit Cosine Similarity: {cosine_similarity_sklearn}")
sentence_vectors = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1]
]);
sentence_similarities = cosine_similarity(sentence_vectors)
print("Cosine Similarity Matrix for Sentences:")
print(sentence_similarities)