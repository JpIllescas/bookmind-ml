"""Carga del dataset semilla desde `seed_data/` (seccion 4.4, paso 1).

Estructura: `manifiesto.csv` (archivo,materia,grado,fuente) + `textos/*.txt`.

No genera texto sintetico: sin corpus real el entrenamiento falla con un
mensaje claro, para que las metricas reportadas signifiquen algo.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .taxonomy import MATERIAS

RUTA_SEMILLA = Path(__file__).resolve().parent.parent / "seed_data"
RUTA_MANIFIESTO = RUTA_SEMILLA / "manifiesto.csv"
RUTA_TEXTOS = RUTA_SEMILLA / "textos"

# Debajo de esto no aporta senal: portada suelta o PDF escaneado.
MINIMO_PALABRAS = 120


@dataclass(frozen=True)
class EjemploEntrenamiento:
    texto: str
    materia: str
    grado: str
    fuente: str
    archivo: str


class DatasetVacioError(RuntimeError):
    """No hay corpus utilizable en seed_data/."""


def cargar_semilla(ruta_manifiesto: Path | None = None) -> list[EjemploEntrenamiento]:
    """Lee el manifiesto y devuelve los ejemplos validos, avisando de lo que descarta."""
    manifiesto = ruta_manifiesto or RUTA_MANIFIESTO

    if not manifiesto.exists():
        raise DatasetVacioError(
            f"No existe el manifiesto en {manifiesto}.\n"
            f"Lee {RUTA_SEMILLA / 'README.md'} para saber como armar el corpus."
        )

    ejemplos: list[EjemploEntrenamiento] = []
    descartados: list[str] = []

    with manifiesto.open(encoding="utf-8-sig", newline="") as f:
        for fila in csv.DictReader(f):
            nombre = (fila.get("archivo") or "").strip()
            materia = (fila.get("materia") or "").strip()

            if not nombre or nombre.startswith("#"):
                continue

            if materia not in MATERIAS:
                descartados.append(f"{nombre}: materia desconocida '{materia}'")
                continue

            ruta_texto = RUTA_TEXTOS / nombre
            if not ruta_texto.exists():
                descartados.append(f"{nombre}: el archivo no existe")
                continue

            texto = ruta_texto.read_text(encoding="utf-8", errors="replace").strip()
            if len(texto.split()) < MINIMO_PALABRAS:
                descartados.append(
                    f"{nombre}: menos de {MINIMO_PALABRAS} palabras "
                    "(extraccion de PDF fallida?)"
                )
                continue

            ejemplos.append(EjemploEntrenamiento(
                texto=texto,
                materia=materia,
                grado=(fila.get("grado") or "").strip(),
                fuente=(fila.get("fuente") or "").strip(),
                archivo=nombre,
            ))

    if descartados:
        print(f"[dataset] {len(descartados)} fila(s) descartada(s):")
        for motivo in descartados:
            print(f"  - {motivo}")

    if not ejemplos:
        raise DatasetVacioError(
            f"El manifiesto {manifiesto} no produjo ningun ejemplo utilizable.\n"
            f"Lee {RUTA_SEMILLA / 'README.md'}."
        )

    return ejemplos


def resumen_por_clase(ejemplos: list[EjemploEntrenamiento]) -> dict[str, int]:
    """Cuenta ejemplos por materia. Sirve para detectar desbalance."""
    conteo: dict[str, int] = {materia: 0 for materia in MATERIAS}
    for ejemplo in ejemplos:
        conteo[ejemplo.materia] += 1
    return {materia: n for materia, n in conteo.items() if n > 0}
