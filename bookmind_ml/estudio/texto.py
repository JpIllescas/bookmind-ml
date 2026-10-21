"""Segmentacion y vocabulario: la base que comparten todos los generadores."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any

# Sin acentos: se compara siempre contra texto normalizado.
STOPWORDS: frozenset[str] = frozenset("""
a al algo alguna algunas alguno algunos ante antes aqui aquel aquella aquellas aquello
aquellos asi aun aunque cada casi como con contra cual cuales cuando cuanto de del
desde donde dos durante e el ella ellas ello ellos en entre era erais eran eras eres es
esa esas ese eso esos esta estaba estaban estamos estan estar estas este esto estos
estoy fue fueron fui ha habia habian han has hasta hay he la las le les lo los luego
mas me mi mia mias mientras mio mios mis mucha muchas mucho muchos muy nada ni no nos
nosotros nuestra nuestras nuestro nuestros nunca o os otra otras otro otros para pero
poco pocos por porque pues que quien quienes se sea sean segun ser si sido siempre sin
sobre solo son soy su sus suya suyas suyo suyos tal tambien tampoco tan tanto te tener
ti tiene tienen toda todas todo todos tu tus un una unas uno unos usted ustedes vez ya
yo vosotros vuestra vuestro dijo dije dice decir habia hacer hace hacia hecho tenia
puede pueden podia podian debe deben entonces despues ahora bien bueno buena asi ahi
alli aca alla cosa cosas manera modo forma parte veces vez dia dias año años tiempo
mismo misma mismos mismas otro otra tras ademas donde cuando mientras ano anos
tengo tienes tenemos tienen tenian voy vas va van vamos iba iban puedo puedes podemos
quiero quieres quiere quieren queria querian sabes sabe saben sabia veo ves vemos vio
vieron hago haces hacen hizo hicieron creo crees cree nadie alguien demasiado bastante
jamas aquello acaso quizas tal vez pronto tarde temprano respondio pregunto exclamo
anadio contesto replico murmuro grito repitio prosiguio continuo observo
uno dos tres cuatro cinco seis siete ocho nueve diez veinte cien mil millones primera
primero segundo segunda tercero tercera ultimo ultima
""".split())

MIN_PALABRAS = 6
MAX_PALABRAS = 60

# Punto, cierre de exclamacion o interrogacion seguido de mayuscula, o un salto de linea.
# "a. C." y "d. C." llevan mayuscula tras el punto y no cierran la oracion.
_RE_LIMITE = re.compile(
    r"(?<=[.!?…])(?<!\ba\.)(?<!\bd\.)\s+(?=[A-ZÁÉÍÓÚÑ¿¡«\"(—\-–])|\n+"
)
_RE_PALABRA = re.compile(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-ZÁÉÍÓÚÜÑ'-]*")
_RE_DIGITO = re.compile(r"\d")
# El romano va en mayusculas estrictas: "vi" o "civil" no son encabezados.
_RE_ENCABEZADO = re.compile(
    r"^(?:[IVXLC]{1,8}|(?i:cap[íi]tulo|unidad|lecci[óo]n|tema|bloque|secci[óo]n|parte)\s+\S+"
    r"(?:\s*[-–:.]\s*(?i:p[áa]g(?:ina)?\.?)\s*\d+)?)\s*[-–:.]?\s+(?=[A-ZÁÉÍÓÚÑ¿¡«\"])"
)


_RE_DIALOGO = re.compile(r"^[—\-–¡¿\"«]|\s—\s?(dijo|preguntó|respondió|añadió|exclamó|contestó)\b")
# Sobre texto normalizado: quien habla de si mismo o a otro no esta definiendo nada.
_RE_PRIMERA_SEGUNDA = re.compile(
    r"\b(yo|me|mi|mis|mio|mia|mios|mias|conmigo|soy|estoy|tu|tus|te|ti|contigo|eres)\b"
)
# Lo que la oracion pide hacer al estudiante en un libro escolar, no lo que explica (yo / nosotros).
_RE_ACTIVIDAD = re.compile(
    r"\b(en (mi|el|tu|su) cuaderno|"
    r"(observ|copi|dibuj|coment|convers|investig|identific|complet|recort|pint|elabor|subray|anot|realiz|"
    r"pregunt|organiz|compar|explic|clasific|dialog|reflexion|solicit|marc|busc|dibuj)(o|amos)|"
    r"(escrib|describ)(o|imos)|(respond|le|hac)(o|emos)|peg(o|amos)|"
    # Instrucciones de juegos de los libros escolares.
    r"el juego consiste|(lanz|tir)(ar|a|an) el dado|su turno|por turnos|gana (quien|el que)|participantes?|"
    r"con el apoyo del maestro|cada uno marca)\b"
)
# Rotulos de ficha escolar: lo que sigue es material de trabajo, no explicacion.
_RE_ROTULO_DE_FICHA = re.compile(r"\b(palabras clave|materiales|instrucciones|procedimiento|hip[oó]tesis)\s*:", re.IGNORECASE)
# Secciones del libro escolar pegadas a la oracion siguiente ("Larios INGUAT Nota de interes En 2008...").
_RE_ROTULO_DE_SECCION = re.compile(
    r"\b(?:Relaciono|Relacionamos|Investigo|Investigamos|Organizo|Organizamos|Taller|Nota de inter[eé]s|"
    r"Actividades adicionales|Evaluaci[oó]n del? proyecto)\b(?=\s+[A-ZÁÉÍÓÚÑ¿¡\d])"
)
# Consigna en "nosotros" al empezar ("Formamos equipos...") o en futuro ("anotaremos"), y codigos de estandar "(CS 4)".
_RE_CONSIGNA_NOSOTROS = re.compile(r"^[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(amos|emos|imos)\b|\b\w+(aremos|eremos|iremos)\b")
_RE_ESTANDAR = re.compile(r"\(\s*[A-Z]{1,3}\s*\d+\s*\)|^Est[aá]ndar\s+\d|\b[Ee]l maestro o (la )?maestra\b")
# Folios o numeros de actividad que quedan delante de la oracion: "6 2 La agricultura es...".
_RE_NUMEROS_INICIALES = re.compile(r"^(?:\d{1,3}\s+)+(?=[A-ZÁÉÍÓÚÑ¿¡])")
# Una cabecera de pagina pegada a la oracion: "MICHEL FOUCAULT: LA ESCALOMETRIA ... Michel Foucault fue..."
MIN_PALABRAS_TITULO = 2


def es_dialogo(texto: str) -> bool:
    return _RE_DIALOGO.search(texto) is not None


def en_primera_o_segunda_persona(texto: str) -> bool:
    return _RE_PRIMERA_SEGUNDA.search(normalizar(texto)) is not None


def es_actividad(texto: str) -> bool:
    """Consignas, juegos, fichas de trabajo y listas con viñetas: no explican el tema."""
    return (
        "•" in texto
        # Una pregunta se la hace el libro al estudiante: no explica nada.
        or "¿" in texto
        or _RE_CONSIGNA_NOSOTROS.search(texto) is not None
        or _RE_ESTANDAR.search(texto) is not None
        or _RE_ROTULO_DE_FICHA.search(texto) is not None
        or _RE_ACTIVIDAD.search(normalizar(texto)) is not None
    )


def _palabra_en_mayusculas(palabra: str) -> bool:
    letras = [c for c in palabra if c.isalpha()]
    # Tolera un error de OCR en palabras largas ("MiCHEL"), pero "El" o "En" no son mayúsculas.
    tolerancia = 1 if len(letras) >= 4 else 0
    return len(letras) >= 2 and sum(not c.isupper() for c in letras) <= tolerancia and letras[0].isupper()


def sin_titulo_inicial(texto: str) -> str:
    """Quita la cabecera en mayusculas que la extraccion pego delante de la oracion."""
    partes = texto.split()
    inicio = 0
    while inicio < len(partes) and (_palabra_en_mayusculas(partes[inicio]) or partes[inicio] in {"Y", "DE", "—", "-", ":"}):
        inicio += 1
    return " ".join(partes[inicio:]) if inicio >= MIN_PALABRAS_TITULO else texto


def es_mayusculas(texto: str) -> bool:
    """Titulos y portadas: casi todas las letras en mayuscula."""
    letras = [c for c in texto if c.isalpha()]
    return len(letras) >= 6 and sum(c.isupper() for c in letras) / len(letras) >= 0.6


# Cabeceras y pies que el libro repite en cada pagina ("9 Guia para docentes de Seminario - Etapa I...").
MIN_PALABRAS_CABECERA = 3
MAX_PALABRAS_CABECERA = 16
FRACCION_CABECERA = 0.3
MIN_PAGINAS_CABECERA = 3
CASI_TODAS = 0.9
# Una cabecera es poco texto dentro de una pagina con contenido: asi no se confunde con oraciones calcadas.
MIN_PALABRAS_PAGINA = 30
MAX_PESO_CABECERA = 0.25


def _es_folio(palabra: str) -> bool:
    return palabra.strip(".-–—·").isdigit()


def _hasta_el_folio(palabras_pagina: list[str]) -> list[str]:
    """El borde de la pagina sin el folio del extremo y hasta el siguiente: el folio separa cabecera y contenido."""
    salida: list[str] = []
    for palabra in palabras_pagina:
        if _es_folio(palabra):
            if salida:
                break
            continue
        salida.append(normalizar(palabra))
    return salida


def _repetida(bordes: list[list[str]]) -> list[str]:
    """La frase del borde (inicio o final) que se repite en mas paginas y, entre esas, la mas larga."""
    if len(bordes) < MIN_PAGINAS_CABECERA:
        return []
    conteo: Counter[tuple[str, ...]] = Counter()
    for borde in bordes:
        for largo in range(MIN_PALABRAS_CABECERA, min(MAX_PALABRAS_CABECERA, len(borde)) + 1):
            conteo[tuple(borde[:largo])] += 1
    minimo = max(MIN_PAGINAS_CABECERA, math.ceil(len(bordes) * FRACCION_CABECERA))
    frecuentes = [(veces, frase) for frase, veces in conteo.items() if veces >= minimo]
    if not frecuentes:
        return []
    # Solo por largo ganaria "cabecera + la primera palabra de algunas paginas".
    tope = max(veces for veces, _ in frecuentes)
    return list(max((len(frase), veces, frase) for veces, frase in frecuentes if veces >= tope * CASI_TODAS)[2])


def _palabras_a_quitar(palabras_pagina: list[str], quitar: list[str]) -> int:
    """Cuantas palabras ocupa la cabecera al inicio (con sus folios intercalados); 0 si no empieza por ella."""
    consumidas = 0
    for i, palabra in enumerate(palabras_pagina):
        # Terminada la frase, el folio pegado a ella tambien se va.
        if consumidas == len(quitar) and not _es_folio(palabra):
            return i
        if _es_folio(palabra):
            continue
        if normalizar(palabra) != quitar[consumidas]:
            return 0
        consumidas += 1
    return len(palabras_pagina) if consumidas == len(quitar) else 0


def _sin_primeras(texto: str, cuantas: int) -> str:
    # Se corta sobre el texto original: los saltos de linea separan oraciones y no deben perderse.
    if not cuantas:
        return texto
    tramos = list(re.finditer(r"\S+", texto))
    return texto[tramos[cuantas].start():] if cuantas < len(tramos) else ""


def _sin_ultimas(texto: str, cuantas: int) -> str:
    if not cuantas:
        return texto
    tramos = list(re.finditer(r"\S+", texto))
    return texto[:tramos[-cuantas - 1].end()] if cuantas < len(tramos) else ""


def _quitar_borde(textos: list[str], final: bool) -> list[str]:
    def palabras_de(texto: str) -> list[str]:
        partes = texto.split()
        return list(reversed(partes)) if final else partes

    # Se detecta solo en paginas con contenido y mirando su primera cuarta parte; se quita de todas.
    con_contenido = [palabras_de(t) for t in textos if len(t.split()) >= MIN_PALABRAS_PAGINA]
    frase = _repetida([_hasta_el_folio(p[:min(MAX_PALABRAS_CABECERA * 2, int(len(p) * MAX_PESO_CABECERA))]) for p in con_contenido])
    if not frase:
        return textos

    salida = []
    for texto in textos:
        partes = palabras_de(texto)
        cuantas = _palabras_a_quitar(partes, frase)
        salida.append(_sin_ultimas(texto, cuantas) if final else _sin_primeras(texto, cuantas))
    return salida


def sin_cabeceras_repetidas(paginas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Quita la cabecera y el pie que el libro repite en muchas paginas, con su numero de pagina."""
    textos = [str(p.get("texto") or "") for p in paginas]
    textos = _quitar_borde(_quitar_borde(textos, final=False), final=True)
    return [{**p, "texto": t} for p, t in zip(paginas, textos)]


@dataclass(frozen=True)
class Oracion:
    texto: str
    pagina: int
    # Posicion global: los generadores devuelven las oraciones en orden de lectura.
    orden: int


def normalizar(texto: str) -> str:
    """Minusculas y sin acentos, para comparar terminos con oraciones."""
    sin_acentos = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in sin_acentos if not unicodedata.combining(c)).lower()


def palabras(texto: str) -> list[str]:
    return _RE_PALABRA.findall(texto)


def tokens(texto: str) -> list[str]:
    """Palabras normalizadas, con stopwords marcadas como cadena vacia para no romper n-gramas."""
    return [
        "" if (t := normalizar(p)) in STOPWORDS or len(t) < 3 else t
        for p in palabras(texto)
    ]


# Igual que el chat del backend (esLegible): piezas reconocibles y, de ellas, al menos la mitad palabras.
MINIMO_LEGIBLE = 0.85
MINIMO_PALABRAS_LEGIBLE = 0.5
_RE_BORDES = re.compile(r"^[^\w]+|[^\w]+$")
_RE_PALABRA_LEGIBLE = re.compile(r"^[^\W\d_]+(?:[-'’][^\W\d_]+)*$")
_RE_NUMERO = re.compile(r"^\d+(?:[.,]\d+)*%?$")


def es_legible(texto: str) -> bool:
    """El OCR de una pagina mala deja "LUSAR ESTOS f TRAJES / Eú": eso no se ensena como si fuera el libro."""
    piezas = [p for p in (_RE_BORDES.sub("", pieza) for pieza in texto.split()) if p]
    if not piezas:
        return False
    palabras_ = sum(
        1 for p in piezas
        if _RE_PALABRA_LEGIBLE.match(p)
        # Solo "a", "e", "o", "u" e "y" son palabras de una letra; las demas, sin vocal, son ruido.
        and (p.lower() in "aeouy" if len(p) == 1 else re.search(r"[aeiouáéíóúüy]", p, re.IGNORECASE))
    )
    numeros = sum(1 for p in piezas if _RE_NUMERO.match(p))
    return (palabras_ + numeros) / len(piezas) >= MINIMO_LEGIBLE and palabras_ / len(piezas) >= MINIMO_PALABRAS_LEGIBLE


def _trozos(texto_pagina: str) -> list[str]:
    """Los trozos de la pagina; la acotacion de un dialogo ("—volvio a preguntar el principito") sigue con su linea."""
    trozos: list[str] = []
    for trozo in _RE_LIMITE.split(texto_pagina):
        limpio = trozo.strip(" \t-–—•·")
        if trozos and trozo.lstrip()[:1] in "-–—" and limpio[:1].islower():
            trozos[-1] = f"{trozos[-1].rstrip()} {trozo.strip()}"
        else:
            trozos.append(trozo)
    return trozos


def segmentar(paginas: list[dict[str, Any]]) -> list[Oracion]:
    """Parte el libro en oraciones con pagina, sin repetidas ni ruido de extraccion."""
    oraciones: list[Oracion] = []
    vistas: set[str] = set()
    orden = 0

    for pagina in sin_cabeceras_repetidas(sorted(paginas, key=lambda p: int(p.get("pagina", 0)))):
        texto_pagina = _RE_ROTULO_DE_SECCION.sub("\n", str(pagina.get("texto") or ""))
        for trozo in _trozos(texto_pagina):
            texto = _RE_NUMEROS_INICIALES.sub("", " ".join(trozo.split()).strip(" \t-–—•·"))
            # El encabezado de capitulo se pega a la primera oracion del capitulo.
            texto = sin_titulo_inicial(_RE_ENCABEZADO.sub("", texto))
            cuantas = len(palabras(texto))

            if not MIN_PALABRAS <= cuantas <= MAX_PALABRAS:
                continue
            # Ejes de graficas, tablas y folios: mas digitos que palabras.
            if len(_RE_DIGITO.findall(texto)) > cuantas:
                continue
            if not es_legible(texto):
                continue

            clave = normalizar(texto)
            if clave in vistas:
                continue

            vistas.add(clave)
            oraciones.append(Oracion(texto, int(pagina.get("pagina", 0)), orden))
            orden += 1

    return oraciones


def contiene(oracion: str, termino: str) -> bool:
    """El termino aparece como palabra completa, sin importar acentos ni mayusculas."""
    patron = r"(?<![a-z0-9])" + re.escape(normalizar(termino)) + r"(?![a-z0-9])"
    return re.search(patron, normalizar(oracion)) is not None


def ocultar(oracion: str, termino: str, hueco: str = "_____") -> str:
    """Reemplaza el termino dentro de la oracion respetando el texto original."""
    normal = normalizar(oracion)
    objetivo = normalizar(termino)
    # Como normalizar conserva la longitud (solo quita marcas), los indices coinciden.
    if len(normal) != len(oracion):
        return oracion.replace(termino, hueco)

    patron = r"(?<![a-z0-9])" + re.escape(objetivo) + r"(?![a-z0-9])"
    salida = oracion
    for coincidencia in reversed(list(re.finditer(patron, normal))):
        salida = salida[: coincidencia.start()] + hueco + salida[coincidencia.end():]
    return salida
