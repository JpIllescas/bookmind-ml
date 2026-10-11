"""Parametros del OCR leidos del entorno, con los valores por defecto del proyecto."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _entero(nombre: str, por_defecto: int, minimo: int = 1) -> int:
    valor = os.environ.get(nombre, "").strip()
    if not valor:
        return por_defecto
    try:
        numero = int(valor)
    except ValueError as error:
        raise ValueError(f"{nombre} debe ser un numero entero; vale '{valor}'.") from error
    if numero < minimo:
        raise ValueError(f"{nombre} debe ser al menos {minimo}; vale {numero}.")
    return numero


# 300 dpi es lo que recomienda Tesseract para texto de libro de 10-12 pt.
OCR_DPI = _entero("OCR_DPI", 300, minimo=72)

# Un hilo libre para que el servicio siga respondiendo /health y /study mientras digitaliza.
OCR_WORKERS = _entero("OCR_WORKERS", max(1, (os.cpu_count() or 2) - 1))

OCR_MAX_PAGINAS = _entero("OCR_MAX_PAGINAS", 600)
OCR_MAX_MB = _entero("OCR_MAX_MB", 150)

# Los trabajos terminados se olvidan pasado este tiempo; el backend ya guardo el resultado.
OCR_TTL_SEGUNDOS = _entero("OCR_TTL_SEGUNDOS", 30 * 60)

OCR_IDIOMA = os.environ.get("OCR_IDIOMA", "spa").strip() or "spa"

CARPETA_TEMPORAL = Path(tempfile.gettempdir()) / "bookmind-ocr"
