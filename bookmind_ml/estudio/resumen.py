"""Resumen extractivo: TextRank sobre similitud TF-IDF y MMR para no repetir ideas."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .texto import STOPWORDS, Oracion, contiene, normalizar, segmentar, tokens

AMORTIGUACION = 0.85
ITERACIONES = 60
# Por debajo de esto dos oraciones no se consideran vecinas en el grafo.
SIMILITUD_MINIMA = 0.05
# Peso de la relevancia frente a la novedad al elegir; 0.7 favorece lo central.
LAMBDA = 0.7
# A partir de estas palabras una oracion ya no se penaliza por corta.
PALABRAS_PLENAS = 14
# Cuanto sube el puntaje por cada concepto del libro que nombra la oracion.
COBERTURA_POR_TERMINO = 0.25
TERMINOS_PARA_RESUMEN = 12
# Palabras con contenido que necesita una oracion para poder representar algo.
MINIMO_CONTENIDO = 4

_RE_DIALOGO = re.compile(r"^[—\-–¡¿\"«]|\s—\s?(dijo|preguntó|respondió|añadió|exclamó|contestó)\b")


def vectorizar(textos: list[str]) -> csr_matrix:
    """TF-IDF normalizado: el producto punto entre filas ya es el coseno."""
    vectorizador = TfidfVectorizer(
        preprocessor=normalizar,
        stop_words=list(STOPWORDS),
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-záéíóúüñ][a-záéíóúüñ]{2,}\b",
    )
    return vectorizador.fit_transform(textos)


def textrank(similitud: np.ndarray) -> np.ndarray:
    """PageRank por iteracion de potencia sobre el grafo de oraciones."""
    n = similitud.shape[0]
    if n == 0:
        return np.array([])

    pesos = similitud.copy()
    np.fill_diagonal(pesos, 0.0)
    pesos[pesos < SIMILITUD_MINIMA] = 0.0

    sumas = pesos.sum(axis=1, keepdims=True)
    # Una oracion sin vecinos reparte su voto entre todas, como en PageRank.
    transicion = np.where(sumas > 0, pesos / np.where(sumas == 0, 1, sumas), 1.0 / n)

    puntaje = np.full(n, 1.0 / n)
    for _ in range(ITERACIONES):
        puntaje = (1 - AMORTIGUACION) / n + AMORTIGUACION * transicion.T @ puntaje

    return puntaje


def mmr(similitud: np.ndarray, relevancia: np.ndarray, cantidad: int, lambda_: float = LAMBDA) -> list[int]:
    """Maximal Marginal Relevance: cada eleccion es relevante y distinta de las anteriores."""
    n = len(relevancia)
    if n == 0:
        return []

    # A la misma escala que la similitud, si no la novedad no pesa nada.
    escala = relevancia.max() or 1.0
    rel = relevancia / escala

    elegidos: list[int] = []
    redundancia = np.zeros(n)

    while len(elegidos) < min(cantidad, n):
        puntaje = lambda_ * rel - (1 - lambda_) * redundancia
        puntaje[elegidos] = -np.inf
        mejor = int(np.argmax(puntaje))
        elegidos.append(mejor)
        redundancia = np.maximum(redundancia, similitud[mejor])

    return elegidos


def _prior(oracion: Oracion) -> float:
    """Una linea de dialogo corta puede ser muy central y aun asi no resumir nada."""
    # "Pregunto a su vez el principito" queda en una sola palabra con contenido: no dice nada.
    if sum(1 for t in tokens(oracion.texto) if t) < MINIMO_CONTENIDO:
        return 0.02

    cuantas = len(oracion.texto.split())
    largo = min(1.0, cuantas / PALABRAS_PLENAS)
    dialogo = 0.6 if _RE_DIALOGO.search(oracion.texto) else 1.0
    return largo * dialogo


def oraciones_centrales(
    oraciones: list[Oracion],
    cantidad: int,
    terminos: list[str] | None = None,
) -> list[Oracion]:
    """Las oraciones mas representativas del conjunto, en orden de lectura."""
    if not oraciones:
        return []

    vectores = vectorizar([o.texto for o in oraciones])
    similitud = (vectores @ vectores.T).toarray()
    puntajes = textrank(similitud) * np.array([_prior(o) for o in oraciones])

    # Nombrar los conceptos del libro vale mas que ser parecida a muchas oraciones.
    if terminos:
        cobertura = np.array([
            sum(1 for t in terminos if contiene(o.texto, t)) for o in oraciones
        ])
        puntajes = puntajes * (1 + COBERTURA_POR_TERMINO * cobertura)

    elegidas = mmr(similitud, puntajes, cantidad)

    return sorted((oraciones[i] for i in elegidas), key=lambda o: o.orden)


def resumir(paginas: list[dict[str, Any]], cantidad: int = 8) -> list[dict[str, Any]]:
    """Puntos del resumen con su pagina, listos para citarse."""
    # Import tardio: conceptos usa oraciones_centrales para elegir contextos.
    from .conceptos import extraer_conceptos

    oraciones = segmentar(paginas)
    terminos = [c.termino for c in extraer_conceptos(paginas, TERMINOS_PARA_RESUMEN)]

    return [
        {"texto": o.texto, "pagina": o.pagina}
        for o in oraciones_centrales(oraciones, cantidad, terminos)
    ]
