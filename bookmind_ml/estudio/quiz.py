"""Quiz de opcion multiple sin LLM: distractores parecidos, pero no iguales, a la respuesta."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from .conceptos import extraer_conceptos
from .flashcards import _oracion_para_cloze
from .resumen import vectorizar
from .texto import contiene, ocultar

OPCIONES = 4
# Distractores con similitud en esta banda: del mismo tema, pero no sinonimos.
BANDA = (0.05, 0.85)


def _semilla(texto: str) -> int:
    # Barajado reproducible: el mismo libro da el mismo quiz.
    return sum(ord(c) for c in texto)


def _distractores(indice: int, similitud: np.ndarray, candidatos: list[str], cuantos: int) -> list[str]:
    """Los mas parecidos dentro de la banda; si no alcanzan, se completa con los demas."""
    orden = np.argsort(-similitud[indice])
    en_banda = [
        int(i) for i in orden
        if i != indice and BANDA[0] <= similitud[indice, i] <= BANDA[1]
    ]
    fuera = [int(i) for i in orden if i != indice and int(i) not in en_banda]

    elegidos: list[str] = []
    for i in [*en_banda, *fuera]:
        if candidatos[i] not in elegidos and candidatos[i] != candidatos[indice]:
            elegidos.append(candidatos[i])
        if len(elegidos) == cuantos:
            break

    return elegidos


def _armar(pregunta: str, correcta: str, distractores: list[str], pagina: int) -> dict[str, Any] | None:
    if len(distractores) < OPCIONES - 1:
        return None

    opciones = [correcta, *distractores[: OPCIONES - 1]]
    random.Random(_semilla(pregunta)).shuffle(opciones)

    return {
        "pregunta": pregunta,
        "opciones": opciones,
        "correcta": opciones.index(correcta),
        "pagina": pagina,
    }


def _recortar(texto: str, maximo: int = 140) -> str:
    if len(texto) <= maximo:
        return texto
    return texto[:maximo].rsplit(" ", 1)[0] + "…"


def generar_quiz(paginas: list[dict[str, Any]], cantidad: int = 5) -> list[dict[str, Any]]:
    # Se piden mas conceptos que preguntas: hacen falta para los distractores.
    conceptos = extraer_conceptos(paginas, max(cantidad * 3, 12))
    if len(conceptos) < OPCIONES:
        return []

    terminos = [c.termino for c in conceptos]
    definidos = [c for c in conceptos if c.es_definicion]

    # Similitud entre terminos por el contexto en que aparecen, no por sus letras.
    contextos = [" ".join(o.texto for o in c.oraciones[:8]) or c.termino for c in conceptos]
    vectores = vectorizar(contextos)
    similitud = (vectores @ vectores.T).toarray()

    preguntas: list[dict[str, Any]] = []
    usadas: set[str] = set()

    # Definiciones y huecos alternados, para variar el tipo de recuerdo que se pide.
    for indice, concepto in enumerate(conceptos):
        if len(preguntas) >= cantidad:
            break

        if concepto.es_definicion and len(definidos) >= OPCIONES:
            otras = [d.definicion for d in definidos if d is not concepto]
            random.Random(_semilla(concepto.termino)).shuffle(otras)
            pregunta = _armar(
                f"¿Cuál de estas afirmaciones describe «{concepto.termino}»?",
                _recortar(concepto.definicion),
                [_recortar(o) for o in otras],
                concepto.pagina,
            )
            if pregunta and pregunta["pregunta"] not in usadas:
                usadas.add(pregunta["pregunta"])
                preguntas.append(pregunta)
                continue

        oracion = _oracion_para_cloze(concepto, usadas)
        if oracion is None or not contiene(oracion.texto, concepto.termino):
            continue

        usadas.add(oracion.texto)
        pregunta = _armar(
            f"¿Qué palabra completa la frase? «{ocultar(oracion.texto, concepto.termino)}»",
            concepto.termino,
            _distractores(indice, similitud, terminos, OPCIONES - 1),
            oracion.pagina,
        )
        if pregunta:
            preguntas.append(pregunta)

    return preguntas[:cantidad]
