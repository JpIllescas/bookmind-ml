"""Tesseract palabra por palabra, para rearmar renglones y medir la confianza de la pagina."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytesseract
from pytesseract import Output

# Ruta del instalador de UB Mannheim, el habitual en Windows; no siempre queda en el PATH.
RUTAS_WINDOWS = (
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
)

# psm 3: segmentacion automatica de la pagina, la adecuada para un libro a una o dos columnas.
CONFIG_TESSERACT = "--psm 3"


class OcrNoDisponibleError(RuntimeError):
    """Tesseract no esta instalado o le falta el idioma pedido."""


@dataclass(frozen=True)
class EstadoTesseract:
    instalado: bool
    version: str | None
    idiomas: tuple[str, ...]

    def tiene(self, idioma: str) -> bool:
        return all(codigo in self.idiomas for codigo in idioma.split("+"))


@dataclass(frozen=True)
class PaginaReconocida:
    texto: str
    confianza: float


def localizar_tesseract() -> str | None:
    comando = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if comando:
        return comando
    return next((str(ruta) for ruta in RUTAS_WINDOWS if ruta.exists()), None)


@lru_cache(maxsize=1)
def estado_tesseract() -> EstadoTesseract:
    """Cacheado: /health se consulta con un timeout de 3 s y lanzar tesseract cuesta."""
    comando = localizar_tesseract()
    if not comando:
        return EstadoTesseract(instalado=False, version=None, idiomas=())

    pytesseract.pytesseract.tesseract_cmd = comando
    try:
        version = str(pytesseract.get_tesseract_version())
        idiomas = tuple(pytesseract.get_languages(config=""))
    except (pytesseract.TesseractNotFoundError, OSError, RuntimeError):
        return EstadoTesseract(instalado=False, version=None, idiomas=())

    return EstadoTesseract(instalado=True, version=version, idiomas=idiomas)


def exigir_tesseract(idioma: str) -> None:
    estado = estado_tesseract()
    if not estado.instalado:
        raise OcrNoDisponibleError(
            "Tesseract no esta instalado en el servidor de ML. Instalalo (en Windows, "
            "el instalador de UB Mannheim) o define TESSERACT_CMD con la ruta del ejecutable."
        )
    if not estado.tiene(idioma):
        raise OcrNoDisponibleError(
            f"Tesseract esta instalado, pero le falta el idioma '{idioma}'. "
            "Reinstalalo marcando el paquete de idioma espanol (spa)."
        )


def reconocer(imagen: np.ndarray, idioma: str) -> PaginaReconocida:
    # Fija la ruta del ejecutable (cacheado) aunque nadie haya consultado /health antes.
    estado_tesseract()
    datos = pytesseract.image_to_data(
        imagen, lang=idioma, config=CONFIG_TESSERACT, output_type=Output.DICT
    )
    return reconstruir(datos)


def reconstruir(datos: dict[str, list[Any]]) -> PaginaReconocida:
    """Une las palabras en renglones y parrafos: la deteccion de capitulos depende de los saltos."""
    renglones: dict[tuple[int, int, int], list[str]] = {}
    confianzas: list[float] = []

    for indice, palabra in enumerate(datos["text"]):
        palabra = str(palabra).strip()
        if not palabra:
            continue

        confianza = float(datos["conf"][indice])
        # -1 marca bloques sin texto reconocido, no palabras con confianza cero.
        if confianza >= 0:
            confianzas.append(confianza)

        clave = (
            int(datos["block_num"][indice]),
            int(datos["par_num"][indice]),
            int(datos["line_num"][indice]),
        )
        renglones.setdefault(clave, []).append(palabra)

    partes: list[str] = []
    anterior: tuple[int, int] | None = None
    for (bloque, parrafo, _), palabras in renglones.items():
        if anterior is not None:
            partes.append("\n\n" if (bloque, parrafo) != anterior else "\n")
        partes.append(" ".join(palabras))
        anterior = (bloque, parrafo)

    media = sum(confianzas) / len(confianzas) / 100 if confianzas else 0.0
    return PaginaReconocida(texto="".join(partes), confianza=round(media, 4))
