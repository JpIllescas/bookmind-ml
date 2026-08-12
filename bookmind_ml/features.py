"""Features interpretables del documento, calculadas sin llamar al LLM."""

from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .readability import analizar
from .taxonomy import (
    LEXICO_POR_MATERIA,
    MATERIAS,
    STOPWORDS_ESPANOL,
    STOPWORDS_INGLES,
)

_RE_PALABRA = re.compile(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+")

# Tamano de segmento para la riqueza lexica (ver `_riqueza_lexica`).
TAMANO_SEGMENTO_MSTTR = 100
_RE_DIGITO = re.compile(r"\d")
# Simbolos que delatan contenido matematico.
_RE_SIMBOLO_MATEMATICO = re.compile(r"[+\-×÷=<>≤≥±%°√∑πΔ/*^]")
# Encabezados tipicos de libro de texto: "Capitulo 3", "Unidad 2", "Leccion 5".
_RE_ENCABEZADO = re.compile(
    r"^\s*(cap[íi]tulo|unidad|lecci[óo]n|tema|bloque|secci[óo]n)\s+"
    r"(\d+|[ivxlcdm]+|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez)\b",
    re.IGNORECASE | re.MULTILINE,
)

# El orden es fijo: el modelo entrenado depende de el.
NOMBRES_FEATURES: tuple[str, ...] = (
    "densidad_matematica",
    "densidad_digitos",
    "prop_stopwords_ingles",
    "prop_stopwords_espanol",
    "riqueza_lexica",
    "largo_promedio_palabra",
    "palabras_por_oracion",
    "silabas_por_palabra",
    "indice_fernandez_huerta",
    "encabezados_por_1000_palabras",
    # Una feature de lexico por cada materia con vocabulario propio.
    *(f"lexico_{materia}" for materia in MATERIAS if LEXICO_POR_MATERIA[materia]),
)


def _riqueza_lexica(palabras: list[str]) -> float:
    """Riqueza lexica (MSTTR): promedia el TTR de segmentos de tamano fijo."""
    n = len(palabras)
    if n == 0:
        return 0.0
    if n < TAMANO_SEGMENTO_MSTTR:
        # Mas corto que un segmento: no hay como normalizar, TTR simple.
        return len(set(palabras)) / n

    segmentos = [
        palabras[i:i + TAMANO_SEGMENTO_MSTTR]
        for i in range(0, n - TAMANO_SEGMENTO_MSTTR + 1, TAMANO_SEGMENTO_MSTTR)
    ]
    return sum(len(set(s)) / len(s) for s in segmentos) / len(segmentos)


def extraer_features(texto: str) -> dict[str, float]:
    """Calcula el vector de features interpretables de un documento."""
    palabras = [p.lower() for p in _RE_PALABRA.findall(texto)]
    n_palabras = len(palabras)
    n_caracteres = max(len(texto), 1)

    legibilidad = analizar(texto)

    if n_palabras == 0:
        return {nombre: 0.0 for nombre in NOMBRES_FEATURES}

    # --- Densidad matematica: simbolos y digitos frente al texto plano ---
    densidad_matematica = len(_RE_SIMBOLO_MATEMATICO.findall(texto)) / n_caracteres
    densidad_digitos = len(_RE_DIGITO.findall(texto)) / n_caracteres

    # --- Deteccion de idioma por stopwords ---
    conjunto_palabras = palabras  # lista, para contar repeticiones
    n_stop_en = sum(1 for p in conjunto_palabras if p in STOPWORDS_INGLES)
    n_stop_es = sum(1 for p in conjunto_palabras if p in STOPWORDS_ESPANOL)

    # --- Vocabulario ---
    riqueza_lexica = _riqueza_lexica(palabras)
    largo_promedio_palabra = sum(len(p) for p in palabras) / n_palabras

    # --- Estructura ---
    encabezados_por_1000 = (len(_RE_ENCABEZADO.findall(texto)) / n_palabras) * 1000

    features: dict[str, float] = {
        "densidad_matematica": densidad_matematica,
        "densidad_digitos": densidad_digitos,
        "prop_stopwords_ingles": n_stop_en / n_palabras,
        "prop_stopwords_espanol": n_stop_es / n_palabras,
        "riqueza_lexica": riqueza_lexica,
        "largo_promedio_palabra": largo_promedio_palabra,
        "palabras_por_oracion": legibilidad.palabras_por_oracion,
        "silabas_por_palabra": legibilidad.silabas_por_palabra,
        "indice_fernandez_huerta": legibilidad.puntuacion,
        "encabezados_por_1000_palabras": encabezados_por_1000,
    }

    # --- Lexico por materia, normalizado por longitud del texto ---
    for materia in MATERIAS:
        vocabulario = LEXICO_POR_MATERIA[materia]
        if not vocabulario:
            continue
        objetivo = set(vocabulario)
        aciertos = sum(1 for p in palabras if p in objetivo)
        features[f"lexico_{materia}"] = (aciertos / n_palabras) * 1000

    return features


class FeaturesNumericas(BaseEstimator, TransformerMixin):
    """Matriz numerica con lo que el TF-IDF no ve: simbolos, legibilidad, frases."""

    def fit(self, X, y=None):  # noqa: N803 - convencion de sklearn
        return self

    def transform(self, X):  # noqa: N803 - convencion de sklearn
        filas = [
            [extraer_features(texto).get(nombre, 0.0) for nombre in NOMBRES_FEATURES]
            for texto in X
        ]
        return np.asarray(filas, dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(NOMBRES_FEATURES, dtype=object)
