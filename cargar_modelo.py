from tflite_runtime.interpreter import Interpreter


def load_classifier(model_path_tflite=None):

    if model_path_tflite is None:
        raise ValueError("Debes especificar un archivo .tflite")

    interpreter = Interpreter(model_path=model_path_tflite)
    interpreter.allocate_tensors()
    return 'tflite', interpreter
