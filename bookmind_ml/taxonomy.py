"""Taxonomia del clasificador; el backend guarda estos strings como enums."""

from __future__ import annotations

# --- Dimension A: materia (la predice el modelo) ---
MATERIAS: tuple[str, ...] = (
    "matematicas",
    "ciencias_naturales",
    "ciencias_sociales",
    "comunicacion_lenguaje",
    "ingles",
    "otro",
)

# --- Dimension B: nivel (formula de legibilidad, no ML) ---
NIVELES: tuple[str, ...] = (
    "primaria_baja",
    "primaria_alta",
    "basicos",
)

# Nombres legibles para la UI y el system prompt.
MATERIA_LEGIBLE: dict[str, str] = {
    "matematicas": "Matemáticas",
    "ciencias_naturales": "Ciencias Naturales",
    "ciencias_sociales": "Ciencias Sociales",
    "comunicacion_lenguaje": "Comunicación y Lenguaje",
    "ingles": "Inglés",
    "otro": "General",
}

NIVEL_LEGIBLE: dict[str, str] = {
    "primaria_baja": "Primaria baja",
    "primaria_alta": "Primaria alta",
    "basicos": "Ciclo básico",
}

# Alimenta la feature de frecuencia lexica por clase.
LEXICO_POR_MATERIA: dict[str, tuple[str, ...]] = {
    "matematicas": (
        "número", "numero", "ecuación", "ecuacion", "suma", "resta",
        "multiplicación", "multiplicacion", "división", "division",
        "geometría", "geometria", "fracción", "fraccion", "problema",
        "triángulo", "triangulo", "área", "area", "perímetro", "perimetro",
        "decimal", "álgebra", "algebra", "operación", "operacion",
    ),
    "ciencias_naturales": (
        "célula", "celula", "energía", "energia", "planta", "experimento",
        "cuerpo", "materia", "animal", "ecosistema", "átomo", "atomo",
        "fuerza", "organismo", "digestión", "digestion", "fotosíntesis",
        "fotosintesis", "mezcla", "temperatura", "sistema", "salud",
    ),
    "ciencias_sociales": (
        "historia", "mapa", "cultura", "gobierno", "fecha", "comunidad",
        "guatemala", "población", "poblacion", "territorio", "municipio",
        "independencia", "constitución", "constitucion", "sociedad",
        "economía", "economia", "derechos", "ciudadano", "maya", "colonial",
    ),
    "comunicacion_lenguaje": (
        "lectura", "gramática", "gramatica", "verbo", "sustantivo", "poema",
        "cuento", "ortografía", "ortografia", "adjetivo", "oración",
        "oracion", "sílaba", "silaba", "acento", "narrador", "personaje",
        "texto", "escritura", "sinónimo", "sinonimo", "párrafo", "parrafo",
    ),
    # El ingles se detecta por stopwords, no por lexico tematico.
    "ingles": (),
    "otro": (),
}

# Frecuentes en ingles y ausentes en espanol: detectan la materia `ingles`.
STOPWORDS_INGLES: frozenset[str] = frozenset({
    "the", "and", "you", "your", "with", "this", "that", "have", "for",
    "are", "what", "they", "from", "will", "can", "about", "there",
    "their", "which", "would", "these", "when", "there", "been", "some",
})

# Contrapeso: con muchas de estas no es un libro de ingles.
STOPWORDS_ESPANOL: frozenset[str] = frozenset({
    "el", "la", "los", "las", "de", "que", "y", "en", "un", "una", "es",
    "por", "con", "para", "se", "del", "al", "lo", "como", "más", "mas",
    "pero", "sus", "le", "ya", "o", "este", "esta", "son", "entre",
})
