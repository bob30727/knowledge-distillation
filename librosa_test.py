import librosa
import numpy as np

# 讀取音檔
audio_path = "test_hello_3.wav"
y, sr = librosa.load(audio_path, sr=None)

# 計算音高 (F0)
# f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=75, fmax=500)
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
f0 = f0[~np.isnan(f0)]
average_pitch = np.mean(f0) if len(f0) > 0 else 0
F0_variance = np.std(f0)
print(f"F0變化量: {F0_variance:.2f} Hz")

# 計算語速（以靜音間隔來估算）
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
print(f"語速: {tempo}")

# 計算能量變化
rms = librosa.feature.rms(y=y)
energy_variance = np.var(rms)
print(f"能量變化: {energy_variance:.6f}")

# print(( np.mean(f0) / 300) + (tempo / 200) + (energy_variance * 10) )
# 快樂程度指數 (0~1)
happiness_score = np.clip((np.std(f0) / 100) + (tempo / 200) + (energy_variance * 10), 0, 2)
print("\n")
print(f"快樂程度分數: {happiness_score.max():.2f}")