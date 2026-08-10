"""Indice Fernandez-Huerta: L = 206.84 - 0.60*P - 1.02*F.

P = silabas por cada 100 palabras, F = promedio de palabras por oracion.
L va de ~0 (muy dificil) a ~100+ (muy facil). Es el nivel de lectura
(dimension B), y no es ML: es una formula determinista.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Vocales del espanol, incluyendo acentuadas y dieresis.
_VOCALES_FUERTES = set("aeoáéóAEOÁÉÓ")
_VOCALES_DEBILES = set("iuíúüIUÍÚÜ")
_VOCALES = _VOCALES_FUERTES | _VOCALES_DEBILES

# Una "palabra" es una secuencia de letras (incluye acentos y n con virgulilla).
_RE_PALABRA = re.compile(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+")
# Fin de oracion: . ! ? ... y sus variantes de apertura del espanol.
_RE_ORACION = re.compile(r"[.!?…]+[\s\"'”»)\]]*")


def contar_silabas(palabra: str) -> int:
    """Cuenta silabas por grupos vocalicos, separando hiatos fuerte-fuerte.

    Heuristica aproximada; sobre miles de palabras el error se promedia.
    """
    palabra = palabra.lower()
    if not palabra:
        return 0

    silabas = 0
    vocal_previa: str | None = None

    for caracter in palabra:
        if caracter in _VOCALES:
            if vocal_previa is None:
                # Arranca un grupo vocalico nuevo.
                silabas += 1
            elif vocal_previa in _VOCALES_FUERTES and caracter in _VOCALES_FUERTES:
                # Hiato: dos vocales fuertes contiguas son silabas distintas
                # ("le-on", "ca-os").
                silabas += 1
            # En cualquier otro caso es diptongo ("cau-sa", "vie-jo"): no suma.
            vocal_previa = caracter
        else:
            vocal_previa = None

    # Toda palabra con letras tiene al menos una silaba (p. ej. "y", "s").
    return max(silabas, 1)


@dataclass(frozen=True)
class MetricasLegibilidad:
    """Resultado del analisis de legibilidad de un texto."""

    puntuacion: float           # indice Fernandez-Huerta
    nivel: str                  # primaria_baja | primaria_alta | basicos
    palabras: int
    oraciones: int
    silabas: int
    silabas_por_palabra: float
    palabras_por_oracion: float


# PENDIENTE: sin calibrar contra el corpus del CNB (lo exige la seccion 1).
UMBRAL_PRIMARIA_BAJA = 80.0   # L >= 80  -> texto muy facil
UMBRAL_PRIMARIA_ALTA = 65.0   # 65 <= L < 80


def clasificar_nivel(puntuacion: float) -> str:
    """Traduce el indice a una de las tres bandas de la taxonomia."""
    if puntuacion >= UMBRAL_PRIMARIA_BAJA:
        return "primaria_baja"
    if puntuacion >= UMBRAL_PRIMARIA_ALTA:
        return "primaria_alta"
    return "basicos"


def analizar(texto: str) -> MetricasLegibilidad:
    """Calcula el indice Fernandez-Huerta y la banda de nivel de un texto."""
    palabras = _RE_PALABRA.findall(texto)
    n_palabras = len(palabras)

    # Sin puntuacion terminal (PDF mal extraido) se cuenta como una oracion.
    n_oraciones = max(len(_RE_ORACION.findall(texto)), 1)

    if n_palabras == 0:
        # Texto vacio o sin letras: devolvemos el nivel mas conservador.
        return MetricasLegibilidad(
            puntuacion=0.0,
            nivel="basicos",
            palabras=0,
            oraciones=0,
            silabas=0,
            silabas_por_palabra=0.0,
            palabras_por_oracion=0.0,
        )

    n_silabas = sum(contar_silabas(p) for p in palabras)

    silabas_por_100 = (n_silabas / n_palabras) * 100
    palabras_por_oracion = n_palabras / n_oraciones

    puntuacion = 206.84 - 0.60 * silabas_por_100 - 1.02 * palabras_por_oracion

    return MetricasLegibilidad(
        puntuacion=round(puntuacion, 2),
        nivel=clasificar_nivel(puntuacion),
        palabras=n_palabras,
        oraciones=n_oraciones,
        silabas=n_silabas,
        silabas_por_palabra=round(n_silabas / n_palabras, 3),
        palabras_por_oracion=round(palabras_por_oracion, 2),
    )
