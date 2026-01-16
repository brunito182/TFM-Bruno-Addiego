import pyaudio
import numpy as np
import threading
import time

p = pyaudio.PyAudio()

# --- Estado del grabador ---
stream = None
frames = []
is_recording = False
lock = threading.Lock()      # <── Protege frames y flags
recording_thread = None


def start_recording(sr=16000):
    global stream, frames, is_recording, recording_thread

    with lock:
        if is_recording:
            return  # ya está grabando

        frames = []  # limpiar buffer
        is_recording = True

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sr,
        input=True,
        frames_per_buffer=1024
    )

    print("🎤 Grabación iniciada")

    def record_thread():
        global frames, is_recording
        while True:
            with lock:
                if not is_recording:
                    break
            data = stream.read(1024, exception_on_overflow=False)
            with lock:
                frames.append(data)
            time.sleep(0.001)

    recording_thread = threading.Thread(target=record_thread)
    recording_thread.start()


def stop_recording(sr=16000):
    global stream, frames, is_recording, recording_thread

    with lock:
        if not is_recording:
            return None, None
        is_recording = False

    # Esperar a que el hilo acabe SIN daemon=True
    if recording_thread is not None:
        recording_thread.join(timeout=0.5)

    try:
        stream.stop_stream()
    except:
        pass
    try:
        stream.close()
    except:
        pass

    print("🛑 Grabación detenida")

    # Bloquear frames antes de leerlos
    with lock:
        audio_bytes = b"".join(frames)
        frames = []  # evitar reuso accidental

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(float)

    print("Tamaño del audio grabado:", len(audio_np))

    if len(audio_np) < 500:
        print("⚠ AUDIO DEMASIADO CORTO — ignorado")
        return None, None

    return audio_np, sr
