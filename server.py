from flask import Flask, jsonify, render_template
from grabador import start_recording, stop_recording
from main_predictor import run_predictor_once_variable_audio
import time

historial = []
ultima_palabra = "--"

app = Flask(__name__)

@app.get("/start")
def start():
    start_recording()
    return jsonify({"status": "grabando", "ultima_palabra": ultima_palabra, "historial": historial})

@app.get("/stop_and_predict")
def stop_and_predict():
    global historial, ultima_palabra

    t0 = time.time()

    audio_array, sr = stop_recording()

    if audio_array is None:
        return jsonify({"prediccion": None, "error": "No se estaba grabando"})

    palabra = run_predictor_once_variable_audio(audio_array, sr)

    latencia = time.time() - t0
    print(f"[PY] Latencia predicción: {latencia:.3f} s")

    historial.insert(0, palabra)
    historial = historial[:5]
    ultima_palabra = palabra

    return jsonify({"prediccion": palabra, "historial": historial, "ultima_palabra": ultima_palabra})

@app.get("/index")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=False)
