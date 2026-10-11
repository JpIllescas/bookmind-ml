"""Deja la pagina como la lee mejor Tesseract: binaria, sin fondo gris y derecha."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

# Un escaneo a mano rara vez se tuerce mas de esto; mas alla suele ser una pagina apaisada.
MAXIMA_INCLINACION = 5.0
PASO_GRADOS = 0.25
# Por debajo de esto rotar solo emborrona la imagen.
MINIMA_CORRECCION = 0.3

# La estimacion se hace en miniatura: el angulo no depende de la resolucion.
ANCHO_ESTIMACION = 800


def preparar(imagen: Image.Image) -> np.ndarray:
    gris = np.asarray(imagen.convert("L"))
    binaria = binarizar(gris)
    angulo = estimar_inclinacion(binaria)
    return rotar(binaria, angulo) if abs(angulo) >= MINIMA_CORRECCION else binaria


def binarizar(gris: np.ndarray) -> np.ndarray:
    # Un desenfoque leve evita que el ruido del papel cuente como tinta.
    suave = cv2.GaussianBlur(gris, (3, 3), 0)
    _, binaria = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaria


def estimar_inclinacion(binaria: np.ndarray) -> float:
    """Angulo (grados) que endereza la pagina: el que deja las filas de texto mas definidas."""
    tinta = 255 - binaria
    if cv2.countNonZero(tinta) < 100:
        return 0.0

    escala = min(1.0, ANCHO_ESTIMACION / tinta.shape[1])
    miniatura = cv2.resize(tinta, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

    angulos = np.arange(-MAXIMA_INCLINACION, MAXIMA_INCLINACION + PASO_GRADOS, PASO_GRADOS)
    puntajes = [_nitidez_de_filas(rotar(miniatura, angulo, fondo=0)) for angulo in angulos]
    return float(angulos[int(np.argmax(puntajes))])


def rotar(imagen: np.ndarray, angulo: float, fondo: int = 255) -> np.ndarray:
    alto, ancho = imagen.shape[:2]
    matriz = cv2.getRotationMatrix2D((ancho / 2, alto / 2), angulo, 1.0)
    return cv2.warpAffine(
        imagen,
        matriz,
        (ancho, alto),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fondo,
    )


def _nitidez_de_filas(tinta: np.ndarray) -> float:
    # Con el texto derecho, las filas alternan entre muy llenas (renglon) y vacias (interlineado).
    perfil = tinta.sum(axis=1, dtype=np.float64)
    return float(np.square(np.diff(perfil)).sum())
