"""Convierte paginas de PDF en imagenes con PDFium (licencia Apache/BSD)."""

from __future__ import annotations

import threading
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

# PDFium no es thread-safe ni siquiera entre documentos distintos: toda llamada pasa por aqui.
CANDADO_PDFIUM = threading.Lock()

PUNTOS_POR_PULGADA = 72


class PdfInvalidoError(ValueError):
    """Los bytes recibidos no se pueden abrir como PDF."""


def contar_paginas(origen: bytes | Path) -> int:
    with CANDADO_PDFIUM:
        documento = _abrir(origen)
        try:
            return len(documento)
        finally:
            documento.close()


def rasterizar(ruta: Path, numero: int, dpi: int) -> Image.Image:
    """Una sola pagina (1-based) en escala de grises; nunca el libro entero en memoria."""
    with CANDADO_PDFIUM:
        documento = _abrir(ruta)
        try:
            pagina = documento[numero - 1]
            try:
                mapa = pagina.render(scale=dpi / PUNTOS_POR_PULGADA, grayscale=True)
                try:
                    # Copia: la imagen de to_pil comparte memoria con el bitmap que se cierra.
                    return mapa.to_pil().convert("L").copy()
                finally:
                    mapa.close()
            finally:
                pagina.close()
        finally:
            documento.close()


def _abrir(origen: bytes | Path) -> pdfium.PdfDocument:
    try:
        return pdfium.PdfDocument(origen if isinstance(origen, bytes) else str(origen))
    except pdfium.PdfiumError as error:
        raise PdfInvalidoError("El archivo no es un PDF valido o esta danado.") from error
