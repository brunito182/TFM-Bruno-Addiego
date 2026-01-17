import numpy as np
import librosa

def preprocessat(audio_array, sr, target_sr=16000):
    # 1. Normalitzar a float32 (-1..1)
    audio = audio_array.astype(np.float32)
    audio /= np.max(np.abs(audio) + 1e-6)

    # 2. Re-mostreig
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 3. Eliminar silencis
    audio, _ = librosa.effects.trim(audio, top_db=30)

    return audio, sr



