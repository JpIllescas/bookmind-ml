"""Extrae texto de un PDF para armar el corpus semilla.

    python scripts/extraer_texto.py libro.pdf seed_data/textos/cnb_mate_3ro.txt

Avisa si el PDF rinde poco texto por pagina: esta escaneado y necesita OCR.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from pypdf import PdfReader

# Debajo de esto la pagina casi seguro es una imagen escaneada.
MINIMO_PALABRAS_POR_PAGINA = 20


def normalizar(texto: str) -> str:
    """Limpia los artefactos tipicos de la extraccion de PDF."""
    # Une palabras cortadas por guion al final de linea: "matemá-\nticas".
    texto = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", texto)
    # Un salto suelto dentro de un parrafo es solo un salto de renglon.
    texto = re.sub(r"(?<![.\n:;])\n(?![\n\s]*[A-ZÁÉÍÓÚÑ0-9])", " ", texto)
    # Colapsa espacios y limita los saltos de parrafo a dos.
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def extraer(ruta_pdf: Path) -> tuple[str, int, int]:
    """Devuelve (texto normalizado, nº de paginas, nº de paginas casi vacias)."""
    lector = PdfReader(str(ruta_pdf))
    partes: list[str] = []
    paginas_vacias = 0

    for pagina in lector.pages:
        contenido = pagina.extract_text() or ""
        if len(contenido.split()) < MINIMO_PALABRAS_POR_PAGINA:
            paginas_vacias += 1
        partes.append(contenido)

    return normalizar("\n\n".join(partes)), len(lector.pages), paginas_vacias


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="PDF de entrada")
    parser.add_argument("salida", type=Path, help="Archivo .txt de salida")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"ERROR: no existe {args.pdf}", file=sys.stderr)
        return 1

    texto, n_paginas, paginas_vacias = extraer(args.pdf)
    n_palabras = len(texto.split())

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(texto, encoding="utf-8")

    print(f"{args.pdf.name}: {n_paginas} páginas -> {n_palabras} palabras")
    print(f"Guardado en {args.salida}")

    if paginas_vacias > n_paginas * 0.5:
        print(
            f"\nAVISO: {paginas_vacias} de {n_paginas} páginas casi no dieron "
            "texto. El PDF probablemente está escaneado y haría falta OCR. "
            "NO lo agregues al manifiesto en este estado.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
