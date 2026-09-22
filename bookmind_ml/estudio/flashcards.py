"""Flashcards sin LLM: termino -> definicion y huecos (cloze) sobre oraciones clave."""

from __future__ import annotations

import re
from typing import Any

from .conceptos import Concepto, extraer_conceptos
from .resumen import oraciones_centrales
from .texto import Oracion, contiene, ocultar

# Una oracion muy larga con un hueco es un acertijo, no una tarjeta.
MAX_PALABRAS_CLOZE = 32


def _pregunta_definicion(concepto: Concepto) -> str:
    termino = concepto.termino
    if not concepto.es_definicion:
        return f"¿Qué dice el libro sobre {termino}?"
    if re.search(r"\b(significa|quiere decir)\b", concepto.definicion, re.IGNORECASE):
        return f"¿Qué significa {termino}?"
    return f"¿Qué es {termino}?"


def _oracion_para_cloze(concepto: Concepto, usadas: set[str]) -> Oracion | None:
    """La oracion mas central entre las que nombran el termino, sin repetir la definicion."""
    candidatas = [
        o for o in concepto.oraciones
        if o.texto not in usadas and len(o.texto.split()) <= MAX_PALABRAS_CLOZE
    ]
    if not candidatas:
        return None

    centrales = oraciones_centrales(candidatas, 1)
    return centrales[0] if centrales else candidatas[0]


def generar_flashcards(paginas: list[dict[str, Any]], cantidad: int = 10) -> list[dict[str, Any]]:
    conceptos = extraer_conceptos(paginas, cantidad)
    tarjetas: list[dict[str, Any]] = []
    usadas: set[str] = set()

    # Se alternan los dos tipos para que la baraja no sea toda definiciones.
    for concepto in conceptos:
        if len(tarjetas) >= cantidad:
            break

        if concepto.definicion not in usadas:
            usadas.add(concepto.definicion)
            tarjetas.append({
                "pregunta": _pregunta_definicion(concepto),
                "respuesta": concepto.definicion,
                "pagina": concepto.pagina,
                "tipo": "definicion" if concepto.es_definicion else "contexto",
            })

        if len(tarjetas) >= cantidad:
            break

        oracion = _oracion_para_cloze(concepto, usadas)
        if oracion is None or not contiene(oracion.texto, concepto.termino):
            continue

        usadas.add(oracion.texto)
        tarjetas.append({
            "pregunta": f"Completa la frase: «{ocultar(oracion.texto, concepto.termino)}»",
            "respuesta": concepto.termino,
            "pagina": oracion.pagina,
            "tipo": "cloze",
        })

    return tarjetas[:cantidad]
