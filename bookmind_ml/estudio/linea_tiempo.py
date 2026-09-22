"""Linea de tiempo: fechas y siglos con la oracion que los contiene, en orden cronologico."""

from __future__ import annotations

import re
from typing import Any

from .texto import segmentar

# Anos de cuatro cifras entre el 1000 y el 2099; "1.000" y cifras sueltas no cuentan.
_RE_ANIO = re.compile(r"(?<![\d.,])(1\d{3}|20\d{2})(?![\d.,])")
_RE_SIGLO = re.compile(r"\bsiglos?\s+([IVXLC]{1,6})\b", re.IGNORECASE)
_RE_ANTES_DE_CRISTO = re.compile(r"\b(\d{1,4})\s*a\.?\s*(?:de\s*)?C\b", re.IGNORECASE)

VALOR_ROMANO = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
MAX_EVENTOS = 14


def _romano(texto: str) -> int | None:
    total = 0
    letras = texto.upper()
    for i, letra in enumerate(letras):
        actual = VALOR_ROMANO.get(letra)
        if actual is None:
            return None
        siguiente = VALOR_ROMANO.get(letras[i + 1], 0) if i + 1 < len(letras) else 0
        total += -actual if actual < siguiente else actual
    return total or None


def _fechas(texto: str) -> list[tuple[int, str]]:
    """(valor para ordenar, etiqueta legible) por cada fecha que mencione la oracion."""
    fechas: list[tuple[int, str]] = []

    for coincidencia in _RE_ANTES_DE_CRISTO.finditer(texto):
        fechas.append((-int(coincidencia.group(1)), f"{coincidencia.group(1)} a. C."))

    for coincidencia in _RE_ANIO.finditer(texto):
        fechas.append((int(coincidencia.group(1)), coincidencia.group(1)))

    for coincidencia in _RE_SIGLO.finditer(texto):
        siglo = _romano(coincidencia.group(1))
        if siglo:
            # Mitad de siglo, para intercalarse bien entre anos concretos.
            fechas.append(((siglo - 1) * 100 + 50, f"Siglo {coincidencia.group(1).upper()}"))

    return fechas


def generar_linea_tiempo(paginas: list[dict[str, Any]], cantidad: int = MAX_EVENTOS) -> list[dict[str, Any]]:
    eventos: dict[tuple[int, int], dict[str, Any]] = {}

    for oracion in segmentar(paginas):
        for valor, etiqueta in _fechas(oracion.texto):
            # Una fecha por pagina: la misma efemeride suele repetirse en el mismo pasaje.
            eventos.setdefault(
                (valor, oracion.pagina),
                {"fecha": etiqueta, "valor": valor, "texto": oracion.texto, "pagina": oracion.pagina},
            )

    ordenados = sorted(eventos.values(), key=lambda e: (e["valor"], e["pagina"]))
    return ordenados[:cantidad]
