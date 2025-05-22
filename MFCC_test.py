# import librosa
# import numpy as np
# import librosa.display
# import matplotlib.pyplot as plt
#
# # 載入音訊
# audio_path = "test_hello_3.wav"
# y, sr = librosa.load(audio_path, sr=22050)
#
# # 計算 MFCC
# mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#
# # 顯示 MFCC 的形狀
# print("MFCC 形狀:", mfcc.shape)  # (13, 時間幀數)
#
# # 繪製 MFCC 熱圖
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(mfcc, x_axis="time", sr=sr)
# plt.colorbar(label="MFCC Coefficients")
# plt.title("MFCC Features")
# plt.xlabel("Time")
# plt.ylabel("MFCC Coefficients")
# plt.show()

###############################################################

# import librosa
# import numpy as np
#
# # 載入音檔
# audio_path = "test_hello_3.wav"  # 替換為你的音檔
# y, sr = librosa.load(audio_path, sr=22050)
#
# # 計算 MFCC 特徵
# mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#
# # 計算 MFCC 第一維度（能量）與變異數
# mfcc_energy = mfcc[0, :]  # 第一個 MFCC 係數代表能量
# energy_mean = np.mean(mfcc_energy)
# energy_var = np.var(mfcc_energy)
#
# # 計算 MFCC 整體變異數（代表聲音變化程度）
# mfcc_var_total = np.mean(np.var(mfcc, axis=1))
#
# # 判斷情緒強度
# if energy_var > 500 and mfcc_var_total > 1000:
#     intensity = "強烈（High Intensity）"
# elif energy_var > 200 and mfcc_var_total > 500:
#     intensity = "中等（Moderate Intensity）"
# else:
#     intensity = "低強度（Low Intensity）"
#
# print(f"MFCC 能量平均值: {energy_mean:.2f}")
# print(f"MFCC 能量變異數: {energy_var:.2f}")
# print(f"MFCC 整體變異數: {mfcc_var_total:.2f}")
# print(f"推測的情緒強度: {intensity}")

###############################################################

import librosa
import numpy as np

# 載入音訊
y, sr = librosa.load("test_hello_1.wav", sr=None)

# 計算 13 個主要 MFCC 係數
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# 計算每個 MFCC 係數的變異數（衡量變化程度）
var_mfcc = np.var(mfcc, axis=1)

# 列印每個係數的變異數
for i, var in enumerate(var_mfcc):
    print(f"MFCC[{i+1}] 變異數: {var:.6f}")

# 設定權重（總和為 1）
weights = np.array([0.5] * 3 + [0.3] * 2 + [0.2] * 8)

# 計算加權總和，作為情緒強度指標
emotion_intensity = np.sum(var_mfcc * weights)

print(f"\n情緒強度指標: {emotion_intensity:.2f}")