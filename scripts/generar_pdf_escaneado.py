"""Convierte un PDF con texto en uno "escaneado": cada pagina pasa a ser solo una imagen.

Sirve para probar el OCR de punta a punta con un libro cuyo texto real se conoce.

    python scripts/generar_pdf_escaneado.py
    python scripts/generar_pdf_escaneado.py --dpi 200 --rotacion 1.5 --ruido 8 --paginas 30
"""

from __future__ import annotations

import argparse
import io
import random
import sys
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

RAIZ = Path(__file__).resolve().parents[2]
ORIGEN = RAIZ / "pdfs-de-prueba" / "Libro_Grande_260_paginas.pdf"
DESTINO = RAIZ / "pdfs-de-prueba" / "Libro_Grande_escaneado.pdf"

PUNTOS_POR_PULGADA = 72
# JPEG como un escaner de oficina: suficiente para el OCR y sin inflar el PDF.
CALIDAD_JPEG = 80


def escanear_pagina(pagina: pdfium.PdfPage, dpi: int, rotacion: float, ruido: float, azar: random.Random) -> Image.Image:
    imagen = pagina.render(scale=dpi / PUNTOS_POR_PULGADA, grayscale=True).to_pil().convert("L")

    if rotacion:
        imagen = imagen.rotate(azar.uniform(-rotacion, rotacion), resample=Image.BICUBIC, fillcolor=255)

    if ruido:
        pixeles = np.asarray(imagen, dtype=np.int16)
        pixeles = pixeles + np.random.default_rng(azar.randrange(2**32)).normal(0, ruido, pixeles.shape)
        imagen = Image.fromarray(np.clip(pixeles, 0, 255).astype(np.uint8))

    return imagen


def generar(origen: Path, destino: Path, dpi: int, rotacion: float, ruido: float, paginas: int | None, semilla: int) -> int:
    azar = random.Random(semilla)
    entrada = pdfium.PdfDocument(str(origen))
    salida = pdfium.PdfDocument.new()
    total = len(entrada) if paginas is None else min(paginas, len(entrada))

    try:
        for indice in range(total):
            pagina = entrada[indice]
            ancho, alto = pagina.get_size()
            imagen = escanear_pagina(pagina, dpi, rotacion, ruido, azar)
            pagina.close()

            jpeg = io.BytesIO()
            imagen.save(jpeg, "JPEG", quality=CALIDAD_JPEG)
            jpeg.seek(0)

            # Pagina por pagina: con save_all de Pillow las 260 imagenes estarian a la vez en memoria.
            nueva = salida.new_page(ancho, alto)
            objeto = pdfium.PdfImage.new(salida)
            objeto.load_jpeg(jpeg, inline=False, autoclose=False)
            objeto.set_matrix(pdfium.PdfMatrix().scale(ancho, alto))
            nueva.insert_obj(objeto)
            nueva.gen_content()
            nueva.close()

            if (indice + 1) % 20 == 0 or indice + 1 == total:
                print(f"  {indice + 1}/{total} paginas", flush=True)

        destino.parent.mkdir(parents=True, exist_ok=True)
        salida.save(str(destino))
    finally:
        salida.close()
        entrada.close()

    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--origen", type=Path, default=ORIGEN)
    parser.add_argument("--destino", type=Path, default=DESTINO)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--rotacion", type=float, default=1.0, help="Grados maximos de inclinacion aleatoria (0 = derecho).")
    parser.add_argument("--ruido", type=float, default=3.0, help="Desviacion del ruido gaussiano en niveles de gris (0 = limpio).")
    parser.add_argument("--paginas", type=int, default=None, help="Solo las primeras N paginas.")
    parser.add_argument("--semilla", type=int, default=7)
    argumentos = parser.parse_args()

    if not argumentos.origen.exists():
        print(f"No existe el PDF de origen: {argumentos.origen}", file=sys.stderr)
        return 1

    print(f"Escaneando {argumentos.origen.name} a {argumentos.dpi} dpi…")
    total = generar(
        argumentos.origen, argumentos.destino, argumentos.dpi,
        argumentos.rotacion, argumentos.ruido, argumentos.paginas, argumentos.semilla,
    )
    tamano = argumentos.destino.stat().st_size / (1024 * 1024)
    print(f"Listo: {argumentos.destino} ({total} paginas, {tamano:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
