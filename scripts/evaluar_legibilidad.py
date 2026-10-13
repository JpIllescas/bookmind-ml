"""Mide la banda de nivel (Fernandez-Huerta) contra el grado real de los libros del Mineduc.

Usa el texto de seed_data/textos/ y el grado de las filas del manifiesto que tienen
un grado concreto. Deja models/metricas_legibilidad.json.

    python scripts/evaluar_legibilidad.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

RAIZ_SERVICIO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_SERVICIO))

from bookmind_ml.readability import (  # noqa: E402
    UMBRAL_PRIMARIA_ALTA,
    UMBRAL_PRIMARIA_BAJA,
    analizar,
    clasificar_nivel,
)

TEXTOS = RAIZ_SERVICIO / "seed_data" / "textos"
MANIFIESTO = RAIZ_SERVICIO / "seed_data" / "manifiesto.csv"
SALIDA = RAIZ_SERVICIO / "models" / "metricas_legibilidad.json"


def banda_esperada(grado: str) -> str:
    # Primero a tercero es primaria baja; cuarto a sexto, primaria alta.
    return "primaria_baja" if int(grado[0]) <= 3 else "primaria_alta"


def libros_con_grado() -> list[tuple[str, str]]:
    with MANIFIESTO.open(encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))
    # Solo con grado concreto (las guias son de todo el ciclo); el recortable no es texto de lectura.
    return [
        (fila["archivo"], fila["grado"])
        for fila in filas
        if (fila["grado"] or "")[:1].isdigit() and "material_manipulable" not in fila["archivo"]
    ]


def main() -> int:
    resultados = []
    for archivo, grado in libros_con_grado():
        metricas = analizar((TEXTOS / archivo).read_text(encoding="utf-8", errors="ignore"))
        esperada = banda_esperada(grado)
        resultados.append({
            "archivo": archivo,
            "grado": grado,
            "puntuacion": metricas.puntuacion,
            "nivel": metricas.nivel,
            "esperado": esperada,
            "acierta": metricas.nivel == esperada,
            # Lo clasico (90/75): para mostrar por que se recalibro.
            "nivelEscalaClasica": "primaria_baja" if metricas.puntuacion >= 90 else
                                  "primaria_alta" if metricas.puntuacion >= 75 else "basicos",
            "silabasPorPalabra": metricas.silabas_por_palabra,
            "palabrasPorOracion": metricas.palabras_por_oracion,
        })

    total = len(resultados)
    salida = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cortes": {"primariaBaja": UMBRAL_PRIMARIA_BAJA, "primariaAlta": UMBRAL_PRIMARIA_ALTA},
        "libros": total,
        "aciertos": sum(r["acierta"] for r in resultados),
        "aciertosEscalaClasica": sum(r["nivelEscalaClasica"] == r["esperado"] for r in resultados),
        # Sin libros de basicos en el corpus, el corte inferior se apoya en un ensayo adulto (56,7) y en el minimo escolar.
        "puntuacionMinimaEscolar": min(r["puntuacion"] for r in resultados),
        "limitacion": "Sin libros de ciclo basico en el corpus; la materia pesa mas que el grado "
                      "(Matematicas 4.o da 88,8 y Lecturas 3.o, 74,8).",
        "porLibro": resultados,
    }
    SALIDA.write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{salida['aciertos']}/{total} con {UMBRAL_PRIMARIA_BAJA:g}/{UMBRAL_PRIMARIA_ALTA:g} "
          f"(escala clasica: {salida['aciertosEscalaClasica']}/{total}). Guardado en {SALIDA}")
    assert clasificar_nivel(salida["puntuacionMinimaEscolar"]) != "basicos"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
