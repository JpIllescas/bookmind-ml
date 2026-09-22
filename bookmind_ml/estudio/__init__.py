"""Motor de estudio propio: genera materiales desde el texto del libro, sin LLM."""

from .conceptos import glosario
from .flashcards import generar_flashcards
from .linea_tiempo import generar_linea_tiempo
from .quiz import generar_quiz
from .resumen import resumir

__all__ = ["glosario", "generar_flashcards", "generar_linea_tiempo", "generar_quiz", "resumir"]
