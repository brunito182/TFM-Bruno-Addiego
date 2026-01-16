import numpy as np
from cargar_modelo import load_classifier
from feature_extraction import detect_speech_segments
from Preprocessat import preprocessat

def predictor_first_stage(
        audio_array,
        sr,
        model_tflite_path,
        target_sr=44100,
        n_mfcc=13,
        target_frames=39,
        top_db=10):

    # 1) Preprocesar audio grabado
    audio, sr = preprocessat(audio_array, sr, target_sr)
    print(f"Audio preprocesado: sr={sr}, muestras={len(audio)}")

    # 2) Cargar modelo TFLite
    model_type, model_obj = load_classifier(model_path_tflite=model_tflite_path)
    print("Modelo cargado:", model_type)

    # 3) Detectar segmentos
    segments = detect_speech_segments(audio, sr, top_db=top_db)
    print("Segmentos detectados:", len(segments))

    palabra = ""
    fonema_anterior = ""

    meta = {
        'model_type': model_type,
        'model': model_obj,
        'audio': audio,
        'sr': sr,
        'segments': segments,
        'n_mfcc': n_mfcc,
        'target_frames': target_frames,
        'top_db': top_db
    }

    return palabra, fonema_anterior, meta, segments
