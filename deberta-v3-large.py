# from transformers import AutoTokenizer, AutoModel
# import torch
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
#
# model_path = "deberta-v3-large"
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
# model = AutoModel.from_pretrained(model_path)
#
# sentences = ["I want to play baseball", "I want to go swimming"]
#
# inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
#
# with torch.no_grad():
#     outputs = model(**inputs)
#
# # embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
# embeddings = outputs.last_hidden_state[:, 0, :]  # 取 CLS token 向量
# embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)  # L2 正規化
#
# # 計算 Cosine 相似度
# # similarity = cosine_similarity([embeddings[0].numpy()], [embeddings[1].numpy()])
# similarity = np.dot(embeddings[0].cpu().numpy(), embeddings[1].cpu().numpy())
#
# print(f"Cosine similarity: {similarity}")


# from transformers import AutoTokenizer, AutoModel
# import torch
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
#
# # 載入 DeBERTa 模型與 tokenizer
# model_path = "deberta-v3-large"
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
# model = AutoModel.from_pretrained(model_path)
#
# # 測試句子
# sentences = ["I want to play baseball", "I want to go swimming"]
#
# # Tokenize
# inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")
#
# # 前向傳播獲取 hidden states
# with torch.no_grad():
#     outputs = model(**inputs)
#
# # 使用 Attention pooling 方法
# attention_mask = inputs['attention_mask'].unsqueeze(-1).float()  # (batch_size, seq_len, 1)
# input_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
#
# # 加權求和
# weighted_sum = torch.sum(input_embeddings * attention_mask, dim=1)  # (batch_size, hidden_dim)
# embedding = weighted_sum / attention_mask.sum(dim=1, keepdim=True)  # (batch_size, hidden_dim)
#
# # L2 正規化
# embedding = embedding / torch.norm(embedding, dim=1, keepdim=True)
#
# # 確保是 (1024,) 而不是 (1, 1024)
# embedding_np_0 = embedding[0].cpu().numpy().reshape(1, -1)  # (1, 1024)
# embedding_np_1 = embedding[1].cpu().numpy().reshape(1, -1)  # (1, 1024)
#
# # 計算 Cosine 相似度
# similarity = cosine_similarity(embedding_np_0, embedding_np_1)
#
# print(f"Cosine similarity: {similarity[0][0]}")



from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

model_path = "deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModel.from_pretrained(model_path)

sentences = ["I want to play baseball", "I want to go swimming"]

inputs = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# 取 CLS token 向量
embeddings = outputs.last_hidden_state[:, 0, :]
embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)  # L2 正規化

# 降維（PCA）
pca = PCA(n_components=50)  # 降至 50 維
embeddings_pca = pca.fit_transform(embeddings.cpu().numpy())

# 計算 Cosine 相似度
similarity = cosine_similarity([embeddings_pca[0]], [embeddings_pca[1]])

print(f"Cosine similarity (after PCA): {similarity[0][0]}")

