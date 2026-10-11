"""Mide el OCR contra el texto real: CER por pagina y global, tiempo y confianza.

Compara el OCR del PDF escaneado (scripts/generar_pdf_escaneado.py) con el texto
nativo del original y deja la evidencia en models/metricas_ocr.json (Fase 5).

    python scripts/evaluar_ocr.py
    python scripts/evaluar_ocr.py --paginas 20
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pypdfium2 as pdfium
from rapidfuzz.distance import Levenshtein

RAIZ_SERVICIO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_SERVICIO))

from bookmind_ml.ocr import configuracion, reconocimiento  # noqa: E402
from bookmind_ml.ocr.trabajos import reconocer_pagina  # noqa: E402

RAIZ = RAIZ_SERVICIO.parent
ORIGINAL = RAIZ / "pdfs-de-prueba" / "Libro_Grande_260_paginas.pdf"
ESCANEADO = RAIZ / "pdfs-de-prueba" / "Libro_Grande_escaneado.pdf"
SALIDA = RAIZ_SERVICIO / "models" / "metricas_ocr.json"


def normalizar(texto: str) -> str:
    # El CER mide caracteres leidos, no el formato: los saltos y espacios dobles no cuentan.
    return re.sub(r"\s+", " ", texto).strip()


def sin_orden(texto: str) -> str:
    # Mismas palabras sin importar el orden: aisla los errores de lectura de los de maquetacion (columnas).
    return " ".join(sorted(texto.split()))


def textos_nativos(ruta: Path, paginas: int) -> list[str]:
    documento = pdfium.PdfDocument(str(ruta))
    try:
        textos = []
        for indice in range(paginas):
            pagina = documento[indice]
            capa = pagina.get_textpage()
            textos.append(normalizar(capa.get_text_bounded()))
            capa.close()
            pagina.close()
        return textos
    finally:
        documento.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original", type=Path, default=ORIGINAL)
    parser.add_argument("--escaneado", type=Path, default=ESCANEADO)
    parser.add_argument("--paginas", type=int, default=None, help="Solo las primeras N paginas.")
    parser.add_argument("--salida", type=Path, default=SALIDA)
    argumentos = parser.parse_args()

    try:
        reconocimiento.exigir_tesseract(configuracion.OCR_IDIOMA)
    except reconocimiento.OcrNoDisponibleError as error:
        print(error, file=sys.stderr)
        return 1

    if not argumentos.escaneado.exists():
        print(
            f"No existe {argumentos.escaneado}. Generalo antes con scripts/generar_pdf_escaneado.py.",
            file=sys.stderr,
        )
        return 1

    escaneado = pdfium.PdfDocument(str(argumentos.escaneado))
    total = len(escaneado) if argumentos.paginas is None else min(argumentos.paginas, len(escaneado))
    escaneado.close()

    nativos = textos_nativos(argumentos.original, total)
    resultados = []
    distancia_total = 0
    distancia_sin_orden_total = 0
    caracteres_total = 0

    print(f"OCR de {total} paginas a {configuracion.OCR_DPI} dpi con '{configuracion.OCR_IDIOMA}'…")
    for numero in range(1, total + 1):
        inicio = time.perf_counter()
        pagina = reconocer_pagina(argumentos.escaneado, numero)
        segundos = time.perf_counter() - inicio

        referencia = nativos[numero - 1]
        obtenido = normalizar(pagina.texto)
        distancia = Levenshtein.distance(obtenido, referencia)
        cer = distancia / len(referencia) if referencia else None
        distancia_sin_orden = Levenshtein.distance(sin_orden(obtenido), sin_orden(referencia))
        cer_sin_orden = distancia_sin_orden / len(referencia) if referencia else None

        if referencia:
            distancia_total += distancia
            distancia_sin_orden_total += distancia_sin_orden
            caracteres_total += len(referencia)

        resultados.append({
            "pagina": numero,
            "cer": round(cer, 4) if cer is not None else None,
            "cerSinOrden": round(cer_sin_orden, 4) if cer_sin_orden is not None else None,
            "confianza": pagina.confianza,
            "segundos": round(segundos, 2),
            "caracteres": len(referencia),
        })
        print(f"  pag. {numero:>3}  CER {cer:6.2%}  sin orden {cer_sin_orden:6.2%}  conf {pagina.confianza:.2f}  {segundos:5.1f}s" if cer is not None
              else f"  pag. {numero:>3}  (sin texto de referencia)")

    medibles = [r for r in resultados if r["cer"] is not None]
    metricas = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "original": argumentos.original.name,
        "escaneado": argumentos.escaneado.name,
        "tesseract": reconocimiento.estado_tesseract().version,
        "idioma": configuracion.OCR_IDIOMA,
        "dpi": configuracion.OCR_DPI,
        "paginas": total,
        # Global ponderado por caracteres: una pagina corta no pesa lo mismo que una llena.
        "cerGlobal": round(distancia_total / caracteres_total, 4) if caracteres_total else None,
        # El CER penaliza leer dos columnas en otro orden aunque cada palabra este bien; este no.
        "cerSinOrdenGlobal": round(distancia_sin_orden_total / caracteres_total, 4) if caracteres_total else None,
        "cerMediana": round(statistics.median(r["cer"] for r in medibles), 4) if medibles else None,
        "confianzaMedia": round(statistics.mean(r["confianza"] for r in resultados), 4) if resultados else None,
        "segundosPorPagina": round(statistics.mean(r["segundos"] for r in resultados), 2) if resultados else None,
        "porPagina": resultados,
    }

    argumentos.salida.parent.mkdir(parents=True, exist_ok=True)
    argumentos.salida.write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"\nCER global {metricas['cerGlobal']:.2%} · sin orden {metricas['cerSinOrdenGlobal']:.2%} · confianza media {metricas['confianzaMedia']:.2f} · "
        f"{metricas['segundosPorPagina']} s/pagina\nGuardado en {argumentos.salida}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
