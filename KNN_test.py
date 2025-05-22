import librosa
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 訓練資料夾
data_dir = "data/audio_samples/"
labels = []
features = []

# 🔹 讀取 WAV 音檔，提取 MFCC 特徵
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)  # 讀取音檔，標準取樣率 16kHz
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 取 13 維 MFCC
    return np.mean(mfccs, axis=1)  # 取平均，轉為 1D 向量

# 🔹 遍歷所有音檔，建立訓練集
for filename in os.listdir(data_dir):
    if filename.endswith(".wav"):
        file_path = os.path.join(data_dir, filename)
        feature_vector = extract_features(file_path)
        features.append(feature_vector)
        labels.append(filename.split("_")[0])  # 假設檔名是 "class1_123.wav"

# 轉換為 NumPy 陣列
X = np.array(features)
y = np.array(labels)

# 🔹 分割訓練集 & 測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 特徵標準化（KNN 受數值範圍影響）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔹 建立 KNN 模型
knn = KNeighborsClassifier(n_neighbors=3)  # 設定 K=3
knn.fit(X_train, y_train)

# 測試模型
accuracy = knn.score(X_test, y_test)
print(f"🎯 KNN 測試準確率: {accuracy:.2f}")

# 🔹 輸入新音訊進行分類
def predict_audio_class(file_path):
    feature_vector = extract_features(file_path).reshape(1, -1)
    feature_vector = scaler.transform(feature_vector)  # 標準化
    prediction = knn.predict(feature_vector)
    return prediction[0]

# 測試新音訊
new_audio = "test_audio.wav"
predicted_class = predict_audio_class(new_audio)
print(f"🎤 輸入音檔分類結果：{predicted_class}")