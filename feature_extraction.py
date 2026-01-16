import numpy as np
import librosa

# 1. Detecció de segments de veu (es retornen intèrvals d'inici i final)

def detect_speech_segments(audio, sr, top_db=30):
    intervals = librosa.effects.split(audio, top_db=top_db)
    return intervals

# 2. Extracció d'MFCC i les seves derivades

def extract_mfcc_with_deltas(
    audio,
    sr,
    n_mfcc=13,
    target_frames=39,
    n_fft=None,
    hop_length=None
):

    # Paràmetres per defecte: 25 ms i hop length de 10 ms.
    if n_fft is None:
        n_fft = int(0.025 * sr)
    if hop_length is None:
        hop_length = int(0.010 * sr)

    # Evitar errors de Librosa si l'àudio és massa curt
    min_len = n_fft * 2
    if len(audio) < min_len:
        pad = min_len - len(audio)
        audio = np.pad(audio, (0, pad), mode="constant")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    n_frames = mfcc.shape[1]

    # Assegurar-se que existeixen 3 o més frames per poder calcular les derivades amb Librosa
    if n_frames < 3:
        mfcc = np.tile(mfcc, (1, int(np.ceil(3 / n_frames))))
        mfcc = mfcc[:, :3]

    # Càlcul de les derivades
    width = min(9, mfcc.shape[1])  # nunca más ancho que el número de frames

    delta = librosa.feature.delta(mfcc, order=1, width=width)
    delta2 = librosa.feature.delta(mfcc, order=2, width=width)

    # Concatenació de derivades
    feat = np.vstack([mfcc, delta, delta2])

    # Ajustar nombre de frames
    n_feat_rows, n_frames = feat.shape

    if n_frames < target_frames:
        pad_width = target_frames - n_frames
        feat = np.pad(feat, ((0, 0), (0, pad_width)), mode="constant")

    elif n_frames > target_frames:
        feat = feat[:, :target_frames]

    # Normalització
    feat = (feat - np.mean(feat)) / (np.std(feat) + 1e-6)

    expected_shape = (3 * n_mfcc, target_frames)
    assert feat.shape == expected_shape, \
        f"Forma inesperada: {feat.shape} (esperado {expected_shape})"

    return feat

