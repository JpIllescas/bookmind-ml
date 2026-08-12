"""Corpus sintetico para tests y demos; el corpus real va en `seed_data/`."""

from __future__ import annotations

import random

PLANTILLAS: dict[str, list[str]] = {
    "matematicas": [
        "Para resolver el problema de suma debemos alinear cada número por su valor posicional.",
        "La fracción representa una parte del entero y se escribe con numerador y denominador.",
        "El área del triángulo se calcula multiplicando la base por la altura y dividiendo entre dos.",
        "Una ecuación es una igualdad entre dos expresiones que contiene un número desconocido.",
        "En geometría estudiamos el perímetro de las figuras planas y sus ángulos.",
        "La multiplicación es una suma repetida del mismo número varias veces.",
        "Ordenamos los números decimales comparando primero la parte entera.",
        "La división reparte una cantidad en grupos iguales sin que sobre resto.",
    ],
    "ciencias_naturales": [
        "La célula es la unidad básica de todos los seres vivos que conocemos.",
        "Las plantas realizan la fotosíntesis para transformar la energía del sol en alimento.",
        "El experimento nos permite comprobar si nuestra hipótesis sobre la materia es correcta.",
        "El cuerpo humano tiene un sistema digestivo que descompone los alimentos.",
        "Un ecosistema reúne a los animales, las plantas y el ambiente donde viven.",
        "La temperatura mide qué tan caliente o frío está un cuerpo.",
        "La fuerza puede cambiar el movimiento o la forma de un objeto.",
        "Cada organismo necesita agua y energía para mantener su salud.",
    ],
    "ciencias_sociales": [
        "La historia de Guatemala incluye el período colonial y la independencia.",
        "El mapa muestra el territorio de cada municipio y su población.",
        "La cultura maya dejó una herencia importante en nuestra comunidad.",
        "El gobierno se organiza según lo que establece la constitución.",
        "La sociedad reconoce los derechos de todo ciudadano sin distinción.",
        "En esa fecha la población celebró un acontecimiento importante.",
        "La economía de la región depende de la agricultura y el comercio.",
        "Cada comunidad conserva tradiciones que forman parte de su identidad.",
    ],
    "comunicacion_lenguaje": [
        "El verbo indica la acción que realiza el sujeto de la oración.",
        "Un sustantivo nombra personas, animales, lugares o cosas.",
        "La lectura del cuento nos ayuda a comprender al personaje principal.",
        "La ortografía correcta requiere colocar el acento en la sílaba tónica.",
        "El poema utiliza versos y un lenguaje lleno de imágenes.",
        "El adjetivo describe una cualidad del sustantivo al que acompaña.",
        "Un párrafo reúne varias oraciones sobre una misma idea.",
        "El narrador cuenta lo que sucede a lo largo del texto.",
    ],
    "ingles": [
        "This is the book that you will use to learn new words and phrases.",
        "What are they doing with these colors and numbers in the classroom?",
        "The teacher will read the story and then you can answer the questions.",
        "There are some animals in the picture and they have different names.",
        "You can practice with your partner about what you like to eat.",
        "These sentences show how the verb changes when we talk about the past.",
        "Would you like to describe your family and where they live?",
        "From this unit you will learn about the weather and the seasons.",
    ],
}


def generar_documento(materia: str, semilla: int, n_oraciones: int = 40) -> str:
    """Arma un documento sintetico repitiendo y barajando las plantillas."""
    rng = random.Random(semilla)
    plantillas = PLANTILLAS[materia]
    oraciones = [rng.choice(plantillas) for _ in range(n_oraciones)]
    # Encabezado para que la feature de estructura tenga senal.
    encabezado = f"Capítulo {rng.randint(1, 8)}\n\n"
    return encabezado + " ".join(oraciones)


def construir_corpus_demo(por_materia: int = 6) -> list[tuple[str, str]]:
    """Devuelve [(texto, materia)] con `por_materia` documentos de cada clase."""
    documentos: list[tuple[str, str]] = []
    for indice, materia in enumerate(PLANTILLAS):
        for k in range(por_materia):
            documentos.append(
                (generar_documento(materia, semilla=indice * 100 + k), materia)
            )
    return documentos
