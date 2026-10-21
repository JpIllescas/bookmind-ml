"""Conceptos del libro: keyphrases por frecuencia y definiciones por patrones."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .resumen import oraciones_centrales
from .texto import (
    Oracion,
    contiene,
    en_primera_o_segunda_persona,
    es_actividad,
    es_dialogo,
    es_mayusculas,
    normalizar,
    palabras,
    segmentar,
    tokens,
)

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
_RESTO_DEBIL = re.compile(
    r"^(muy|tan|m[aá]s|menos|casi|s[oó]lo|bastante|demasiado|un poco|as[ií])\b", re.IGNORECASE
)

# Con "es" o "son" cualquier frase parece definicion; solo cuentan para terminos frecuentes.
_VERBOS_DEBILES = {"es", "son", "era", "eran", "comprende", "abarca", ""}

# "La celula es la unidad..." clasifica; "El desierto es bello" solo describe.
_VERBOS_SINGULARES = {"es", "era"}
_PREDICADO_QUE_CLASIFICA = re.compile(
    r"^(el|la|lo|los|las|un|una|unos|unas|uno de|una de|aquel|aquella|aquello)\b", re.IGNORECASE
)

# Rotulos de los libros escolares ("Importante: los nombres no se repiten"): anuncian, no son un termino.
_ROTULOS = frozenset("""
importante nota recuerda recuerdo observa observo ejemplo ejemplos atencion actividad actividades pista consejo
aprendo aprendemos glosario resumen conclusion conclusiones respuesta respuestas solucion pregunta preguntas
reflexiono reflexionamos sabias que dato curioso para saber mas tarea instrucciones
""".split())

# "Materiales: un limón, ruda, tilo..." es una lista, no una definicion.
_PATRON_DOS_PUNTOS = _PATRONES_DEFINICION[2]
PALABRAS_POR_ELEMENTO_DE_LISTA = 3


def _es_lista(resto: str) -> bool:
    elementos = [e for e in re.split(r",|\by\b", resto) if e.strip()]
    return len(elementos) >= 3 and sum(len(e.split()) for e in elementos) / len(elementos) <= PALABRAS_POR_ELEMENTO_DE_LISTA


@dataclass
class Concepto:
    termino: str
    definicion: str
    pagina: int
    veces: int
    # True si la definicion sale de un patron ("X es..."); si no, es la oracion mas central.
    es_definicion: bool
    oraciones: list[Oracion] = field(default_factory=list)
    # Varias apariciones distintas: con ellas Gemini puede decir que es el termino, no solo citarlo.
    contextos: list[Oracion] = field(default_factory=list)


CONTEXTOS_POR_TERMINO = 4
# Palabras del comienzo donde el termino suele ser el tema de la oracion y no un detalle.
PALABRAS_DEL_COMIENZO = 6
COMAS_DE_UNA_LISTA = 3


def _explica(oracion: Oracion) -> bool:
    """Narra o explica: no es dialogo, ni lista, ni consigna, ni un trozo que empieza a media oracion."""
    texto = oracion.texto
    return not (
        es_dialogo(texto)
        or en_primera_o_segunda_persona(texto)
        or es_actividad(texto)
        or texto[:1].islower()
        or texto.count(",") >= COMAS_DE_UNA_LISTA
    )


def _contexto(apariciones: list[Oracion], termino: str) -> Oracion:
    """Sin definicion, la oracion que mejor lo presenta: que explique y, si se puede, que lo tenga de tema."""
    def de_tema(oracion: Oracion) -> bool:
        return contiene(" ".join(palabras(oracion.texto)[:PALABRAS_DEL_COMIENZO]), termino)

    for elegibles in (
        [o for o in apariciones if _explica(o) and de_tema(o)],
        [o for o in apariciones if _explica(o)],
        apariciones,
    ):
        if elegibles:
            return oraciones_centrales(elegibles, 1)[0]
    return apariciones[0]


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
    # "Admirar significa reconocer que yo soy..." o "—añadió el principito": es un personaje hablando.
    if es_dialogo(oracion) or en_primera_o_segunda_persona(oracion) or es_actividad(oracion):
        return None

    for patron in _PATRONES_DEFINICION:
        coincidencia = patron.match(oracion.strip())
        if not coincidencia:
            continue

        termino = coincidencia.group("termino").strip(" \"«»'")
        resto = coincidencia.group("resto").strip()
        verbo = (coincidencia.groupdict().get("verbo") or "").lower()

        if len(palabras(termino)) > 4 or _RESTO_DEBIL.match(resto):
            continue
        # "ERNST JUNGER: PODER Y..." es un titulo; "número 2 era así" no es un termino.
        if es_mayusculas(termino) or re.search(r"\d", termino):
            continue
        if verbo in _VERBOS_SINGULARES and not _PREDICADO_QUE_CLASIFICA.match(resto):
            continue
        if patron is _PATRON_DOS_PUNTOS and (_es_lista(resto) or normalizar(termino) in _ROTULOS):
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
        ((clave, veces) for clave, veces in frecuencia.items() if veces >= MIN_FRECUENCIA and clave not in _ROTULOS),
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
        # "Intercambio mi cuaderno con…": lo que el libro solo nombra en consignas o diálogos no es un concepto suyo.
        if clave not in definidos and not any(_explica(o) for o in apariciones):
            continue

        if clave in definidos:
            # Vino de un patron ("X es...", "X consiste en..."): la fuerza del verbo solo ordena.
            _, definicion, pagina, _ = definidos[clave]
            es_definicion = True
        else:
            contexto = _contexto(apariciones, termino)
            definicion, pagina, es_definicion = contexto.texto, contexto.pagina, False

        # Las que explican van primero; MMR las elige distintas entre si.
        explicativas = [o for o in apariciones if _explica(o)] or apariciones
        contextos = oraciones_centrales(explicativas, CONTEXTOS_POR_TERMINO) if explicativas else []

        conceptos.append(
            Concepto(
                termino=termino if termino[:1].isupper() else termino.lower(),
                definicion=definicion,
                pagina=pagina,
                veces=max(len(apariciones), frecuencia.get(clave, 0)),
                es_definicion=es_definicion,
                oraciones=apariciones,
                contextos=contextos,
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
            "contextos": [{"texto": o.texto, "pagina": o.pagina} for o in c.contextos],
        }
        for c in extraer_conceptos(paginas, cantidad)
    ]
