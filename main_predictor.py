import time
from predictor_first_stage import predictor_first_stage
from predictor_core import predictor_core
from jaccard_distance import corregir_palabra
from grabador import start_recording, stop_recording

MODEL_TFLITE_PATH = "/home/bruno/Desktop/classificador_fonemes.tflite"
WORDS_FILE = "/home/bruno/Desktop/palabras.txt"

def run_predictor_once_variable_audio(audio_array, sr):
    
    t0 = time.perf_counter()
    
    t1 = time.perf_counter()
    palabra, fonema_anterior, meta, segments = predictor_first_stage(
        audio_array=audio_array,
        sr=sr,
        model_tflite_path=MODEL_TFLITE_PATH
    )
    
    t2 = time.perf_counter()

    palabra_predicha = predictor_core(
        audio=meta['audio'],
        sr=meta['sr'],
        segments=segments,
        model=meta['model'],
        target_frames=meta['target_frames']
    )
    
    t3 = time.perf_counter()

    palabra_final = corregir_palabra(palabra_predicha, WORDS_FILE)
    
    t4 = time.perf_counter()
    
    print("TIMINGS:")
    print(f"  Preprocessat      : {t2 - t1:.3f} s")
    print(f"  Cos               : {t3 - t2:.3f} s")
    print(f"  Correcció final   : {t4 - t3:.3f} s")
    print(f"  TOTAL             : {t4 - t0:.3f} s\n")
    
    return palabra_final

if __name__ == "__main__":
    print(run_predictor_once())
