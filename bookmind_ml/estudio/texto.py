"""Segmentacion y vocabulario: la base que comparten todos los generadores."""

from __future__ import annotations

import re
import unicodedata
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
    r"pregunt|organiz|compar|explic|clasific|dialog|reflexion)(o|amos)|"
    r"(escrib|describ)(o|imos)|(respond|le)(o|emos)|peg(o|amos))\b"
)
# Una cabecera de pagina pegada a la oracion: "MICHEL FOUCAULT: LA ESCALOMETRIA ... Michel Foucault fue..."
MIN_PALABRAS_TITULO = 2


def es_dialogo(texto: str) -> bool:
    return _RE_DIALOGO.search(texto) is not None


def en_primera_o_segunda_persona(texto: str) -> bool:
    return _RE_PRIMERA_SEGUNDA.search(normalizar(texto)) is not None


def es_actividad(texto: str) -> bool:
    """Consignas para el estudiante y listas con viñetas: no explican el tema."""
    return "•" in texto or _RE_ACTIVIDAD.search(normalizar(texto)) is not None


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


def segmentar(paginas: list[dict[str, Any]]) -> list[Oracion]:
    """Parte el libro en oraciones con pagina, sin repetidas ni ruido de extraccion."""
    oraciones: list[Oracion] = []
    vistas: set[str] = set()
    orden = 0

    for pagina in sorted(paginas, key=lambda p: int(p.get("pagina", 0))):
        for trozo in _RE_LIMITE.split(str(pagina.get("texto") or "")):
            texto = " ".join(trozo.split()).strip(" \t-–—•·")
            # El encabezado de capitulo se pega a la primera oracion del capitulo.
            texto = sin_titulo_inicial(_RE_ENCABEZADO.sub("", texto))
            cuantas = len(palabras(texto))

            if not MIN_PALABRAS <= cuantas <= MAX_PALABRAS:
                continue
            # Ejes de graficas, tablas y folios: mas digitos que palabras.
            if len(_RE_DIGITO.findall(texto)) > cuantas:
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
