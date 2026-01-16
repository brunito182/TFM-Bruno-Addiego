import pandas as pd

def jaccard_distance(str1: str, str2: str) -> float:

    set1 = set(str1)
    set2 = set(str2)
    interseccion = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 1.0  # Evita divisions entre 0
    return 1 - interseccion / union


def corregir_palabra(palabra_predicha: str, archivo_palabras: str = "palabras.txt") -> str:

    # Llegir la llista de paraules des del .txt
    tabla = pd.read_csv(archivo_palabras, sep="\t", header=None, names=["Palabras"])
    
    distancia_minima = float("inf")
    palabra_correcta = ""

    for _, fila in tabla.iterrows():
        palabra_ref = fila["Palabras"]
        distancia = jaccard_distance(palabra_predicha, palabra_ref)
        if distancia < distancia_minima:
            distancia_minima = distancia
            palabra_correcta = palabra_ref

    return palabra_correcta