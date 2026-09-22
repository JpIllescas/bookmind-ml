"""Conceptos del libro: keyphrases por frecuencia y definiciones por patrones."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .resumen import oraciones_centrales
from .texto import Oracion, contiene, palabras, segmentar, tokens

MAX_NGRAMA = 3
MIN_FRECUENCIA = 2
# Cuantos candidatos se conservan antes de buscarles definicion.
CANDIDATOS = 60

_ARTICULO = r"(?:el |la |los |las |un |una |unos |unas )?"

# "La celula es la unidad basica...", "Domesticar significa crear lazos", "...se llama fotosintesis".
_PATRONES_DEFINICION = [
    re.compile(
        r"^" + _ARTICULO + r"(?P<termino>[^,;:()]{2,45}?)\s+"
        r"(?P<verbo>es|son|era|eran|consiste en|significa|quiere decir|se define como|"
        r"se denomina|se llama|se refiere a|se conoce como|se entiende por|comprende|abarca)\s+"
        r"(?P<resto>\S.{15,})$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<resto>.{15,}?)\b(?:se (?:llama|denomina|conoce como)|llamad[oa]s?|denominad[oa]s?|"
        r"conocid[oa]s? como)\s+" + _ARTICULO + r"(?P<termino>[^,;:().]{2,45})\.?$",
        re.IGNORECASE,
    ),
    re.compile(r"^(?P<termino>[^:]{2,45}):\s+(?P<resto>\S.{15,})$"),
]

# Verbos que no definen sino que narran: "el principito es muy pequeno" no es definicion.
_RESTO_DEBIL = re.compile(r"^(muy|tan|mas|menos|casi|solo|bastante|demasiado|un poco)\b", re.IGNORECASE)

# Con "es" o "son" cualquier frase parece definicion; solo cuentan para terminos frecuentes.
_VERBOS_DEBILES = {"es", "son", "era", "eran", "comprende", "abarca", ""}


@dataclass
class Concepto:
    termino: str
    definicion: str
    pagina: int
    veces: int
    # True si la definicion sale de un patron ("X es..."); si no, es la oracion mas central.
    es_definicion: bool
    oraciones: list[Oracion] = field(default_factory=list)


def _candidatos(oraciones: list[Oracion]) -> tuple[Counter[str], dict[str, str]]:
    """N-gramas de palabras con contenido; devuelve frecuencias y la forma mas comun."""
    frecuencia: Counter[str] = Counter()
    formas: dict[str, Counter[str]] = defaultdict(Counter)

    for oracion in oraciones:
        crudas = palabras(oracion.texto)
        limpias = tokens(oracion.texto)

        for n in range(1, MAX_NGRAMA + 1):
            for i in range(len(limpias) - n + 1):
                ventana = limpias[i:i + n]
                # Un n-grama con stopwords dentro ("de", "el") no es un termino.
                if any(t == "" for t in ventana):
                    continue

                clave = " ".join(ventana)
                frecuencia[clave] += 1
                formas[clave][" ".join(crudas[i:i + n]).lower()] += 1

    return frecuencia, {clave: conteo.most_common(1)[0][0] for clave, conteo in formas.items()}


def _sin_solapes(ranking: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Si "fotosintesis" solo aparece dentro de "proceso de fotosintesis", se queda el largo."""
    elegidos: list[tuple[str, int]] = []
    for clave, veces in ranking:
        redundante = any(
            (clave in otra or otra in clave) and abs(otras_veces - veces) <= veces * 0.4
            for otra, otras_veces in elegidos
        )
        if not redundante:
            elegidos.append((clave, veces))
    return elegidos


def _definicion(oracion: str) -> tuple[str, str, bool] | None:
    """Termino, oracion completa y si el verbo define de verdad ("significa", "se llama")."""
    for patron in _PATRONES_DEFINICION:
        coincidencia = patron.match(oracion.strip())
        if not coincidencia:
            continue

        termino = coincidencia.group("termino").strip(" \"«»'")
        resto = coincidencia.group("resto").strip()
        verbo = (coincidencia.groupdict().get("verbo") or "").lower()

        if len(palabras(termino)) > 4 or _RESTO_DEBIL.match(resto):
            continue
        # "En mi tierra" o "Y añadió" no son terminos: deben abrir y cerrar con palabra de contenido.
        partes = tokens(termino)
        if not partes or partes[0] == "" or partes[-1] == "":
            continue

        return termino, oracion.strip(), verbo not in _VERBOS_DEBILES

    return None


def extraer_conceptos(paginas: list[dict[str, Any]], cantidad: int = 12) -> list[Concepto]:
    oraciones = segmentar(paginas)
    if not oraciones:
        return []

    frecuencia, forma = _candidatos(oraciones)

    # Los multipalabra pesan mas: "sistema solar" dice mas que "sistema".
    ranking = sorted(
        ((clave, veces) for clave, veces in frecuencia.items() if veces >= MIN_FRECUENCIA),
        key=lambda par: (par[1] * (1 + 0.5 * (par[0].count(" "))), par[0]),
        reverse=True,
    )
    candidatos = dict(_sin_solapes(ranking[: CANDIDATOS * 3])[:CANDIDATOS])

    definidos: dict[str, tuple[str, str, int, bool]] = {}
    for oracion in oraciones:
        resultado = _definicion(oracion.texto)
        if not resultado:
            continue

        termino, definicion, fuerte = resultado
        clave = " ".join(t for t in tokens(termino) if t)
        if not clave or clave in definidos:
            continue
        # "X es ..." solo cuenta si X es un termino que el libro repite.
        if not fuerte and clave not in candidatos:
            continue

        definidos[clave] = (termino, definicion, oracion.pagina, fuerte)

    # Primero lo definido de verdad, luego lo enunciado ("X es...") y al final lo frecuente.
    claves = list(dict.fromkeys([
        *(c for c, d in definidos.items() if d[3]),
        *(c for c, d in definidos.items() if not d[3]),
        *candidatos.keys(),
    ]))

    conceptos: list[Concepto] = []
    for clave in claves:
        termino = definidos[clave][0] if clave in definidos else forma[clave]
        apariciones = [o for o in oraciones if contiene(o.texto, termino)]
        if clave not in definidos and len(apariciones) < MIN_FRECUENCIA:
            continue

        if clave in definidos:
            _, definicion, pagina, es_definicion = definidos[clave]
        else:
            # Sin definicion: la oracion mas central entre las que lo nombran hace de contexto.
            contexto = oraciones_centrales(apariciones, 1)[0]
            definicion, pagina, es_definicion = contexto.texto, contexto.pagina, False

        conceptos.append(
            Concepto(
                termino=termino if termino[:1].isupper() else termino.lower(),
                definicion=definicion,
                pagina=pagina,
                veces=max(len(apariciones), frecuencia.get(clave, 0)),
                es_definicion=es_definicion,
                oraciones=apariciones,
            )
        )
        if len(conceptos) >= cantidad:
            break

    return conceptos


def glosario(paginas: list[dict[str, Any]], cantidad: int = 12) -> list[dict[str, Any]]:
    return [
        {
            "termino": c.termino,
            "definicion": c.definicion,
            "pagina": c.pagina,
            "veces": c.veces,
            "esDefinicion": c.es_definicion,
        }
        for c in extraer_conceptos(paginas, cantidad)
    ]
