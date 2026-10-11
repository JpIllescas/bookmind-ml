"""Arregla los artefactos tipicos del OCR sin inventar texto: nada de corrector ortografico."""

from __future__ import annotations

import re
import unicodedata

LIGADURAS = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}

# Guion, guion corto unicode, guion no separable y el signo de corte que a veces sale del OCR.
GUION_FIN_DE_RENGLON = re.compile(r"([^\W\d_])[-‐‑­¬][ \t]*\n[ \t]*([^\W\d_])")

ESPACIOS = re.compile(r"[ \t  -​ 　]+")

# Por debajo de esta proporcion de letras, un renglon es ruido: bordes, manchas o numeros sueltos.
MINIMO_LETRAS = 0.4


def limpiar(texto: str) -> str:
    texto = unicodedata.normalize("NFC", texto)
    for ligadura, letras in LIGADURAS.items():
        texto = texto.replace(ligadura, letras)

    texto = unir_palabras_cortadas(texto)

    renglones = []
    for renglon in texto.split("\n"):
        renglon = ESPACIOS.sub(" ", renglon).strip()
        if renglon and es_basura(renglon):
            continue
        renglones.append(renglon)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(renglones)).strip()


def unir_palabras_cortadas(texto: str) -> str:
    """'cé-\\nlula' -> 'célula'; un guion seguido de mayuscula es compuesto ('Norte-Sur') y se respeta."""

    def unir(coincidencia: re.Match[str]) -> str:
        antes, despues = coincidencia.group(1), coincidencia.group(2)
        if despues.isupper():
            return f"{antes}-{despues}"
        return antes + despues

    return GUION_FIN_DE_RENGLON.sub(unir, texto)


def es_basura(renglon: str) -> bool:
    visibles = [caracter for caracter in renglon if not caracter.isspace()]
    if not visibles:
        return False
    letras = sum(caracter.isalpha() for caracter in visibles)
    return letras / len(visibles) < MINIMO_LETRAS
