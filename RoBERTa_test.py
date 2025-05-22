from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Do you want to go swimming?
# No, I feel like playing basketball today.
# Yeah, the weather looks pretty good today.
context = "Do you want to go swimming?"
user_response = "Yeah, the weather looks pretty good today."
full_input = f"{context}\n{user_response}"
sentences = [full_input,
             "I want to go swimming"]

model = SentenceTransformer("nli-roberta-base-v2")
# all_roberta_large_v1 (0.68-0.70)
# all-MiniLM-L6-v2 (0.81-0.80)
# stsb-roberta-large (0.51-0.61)
# all-mpnet-base-v2 (0.68-0.73)
# paraphrase-MiniLM-L6-v2 (都很高)
# nli-roberta-base-v2 (0.58-0.74)

start_time = time.time()
embeddings = model.encode(sentences)

# 計算兩個句子的餘弦相似度
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")

# 輸出相似度
print(sentences[0], "\n")
print(sentences[1], "\n")
print(f"Cosine similarity: {similarity[0][0]}")




