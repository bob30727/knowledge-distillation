import librosa
import librosa.display
import numpy as np

import librosa
import numpy as np

# 讀取音訊
wav_path = "example.wav"
audio, sr = librosa.load(wav_path, sr=None)

# 計算能量（Energy）來偵測語音區段
energy = librosa.feature.rms(y=audio)[0]
threshold = np.percentile(energy, 75)  # 設定閾值來判斷是否有語音
speech_frames = np.where(energy > threshold)[0]

# 計算語音的總時間
speech_duration = len(speech_frames) / sr  # 語音區間的總秒數

# 假設每秒平均講 4 個詞（這可根據語料庫調整）
words_per_second = 4  
wpm = words_per_second * (speech_duration / 60)

print(f"語速 (WPM)：{wpm:.2f} words per minute")


# 讀取音訊
audio, sr = librosa.load("example.wav", sr=None)

# 使用 PYIN 算法來擷取音高 (F0)
f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=50, fmax=500, sr=sr)

# 移除 NaN 值（因為某些區段可能偵測不到音高）
f0 = f0[~np.isnan(f0)]

# 計算平均音高
mean_pitch = np.mean(f0)
print(f"平均音高 (Pitch) : {mean_pitch:.2f} Hz")