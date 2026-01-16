import numpy as np
import time
from tflite_runtime.interpreter import Interpreter
from feature_extraction import extract_mfcc_with_deltas

# Cargar los fonemas
with open("fonemas.txt", "r", encoding="utf-8") as f:
    FONEMAS = [line.strip() for line in f if line.strip()]


def predictor_core(audio, sr, segments, model, target_frames=39):

    t_core_start = time.perf_counter()

    t_mfcc = 0.0
    t_infer = 0.0
    n_windows = 0

    fonemas_detectados = []

    for seg_idx, (start, end) in enumerate(segments):
        segmento = audio[start:end]
        seg_len = len(segmento)

        if seg_len < int(0.02 * sr):
            continue

        win_factor = 0.3
        win_length = int(win_factor * seg_len)
        hop = win_length // 2

        if win_length < int(0.02 * sr):
            win_length = int(0.02 * sr)
        if hop < 1:
            hop = 1

        for i in range(0, seg_len - win_length + 1, hop):
            window = segmento[i:i + win_length]
            if np.all(window == 0):
                continue

            # ---------- MFCC ----------
            t0 = time.perf_counter()
            try:
                feat = extract_mfcc_with_deltas(
                    window, sr, n_mfcc=13, target_frames=target_frames
                )
            except:
                continue
            t_mfcc += time.perf_counter() - t0

            feat = np.expand_dims(feat, axis=(0, -1)).astype(np.float32)

            # ---------- INFERENCIA ----------
            t0 = time.perf_counter()
            try:
                input_details = model.get_input_details()
                output_details = model.get_output_details()
                model.set_tensor(input_details[0]['index'], feat)
                model.invoke()
                pred = model.get_tensor(output_details[0]['index'])[0]
            except:
                continue
            t_infer += time.perf_counter() - t0

            fonema_idx = int(np.argmax(pred))
            fonema = FONEMAS[fonema_idx] if fonema_idx < len(FONEMAS) else f"UNK{fonema_idx}"

            fonemas_detectados.append(fonema)
            n_windows += 1

    # ---------- POST ----------
    if not fonemas_detectados:
        palabra = "Desconocido"
    else:
        fonemas_filtrados = [fonemas_detectados[0]]
        for f in fonemas_detectados[1:]:
            if f != fonemas_filtrados[-1]:
                fonemas_filtrados.append(f)
        palabra = "".join(fonemas_filtrados)

    t_total = time.perf_counter() - t_core_start

    # ---------- LOG ----------
    print(
        f"[TIMING predictor_core] total={t_total:.3f}s | "
        f"mfcc={t_mfcc:.3f}s | infer={t_infer:.3f}s | "
        f"windows={n_windows}"
    )

    return palabra
