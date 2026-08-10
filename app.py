"""Microservicio de clasificacion — FastAPI (seccion 4.3).

    uvicorn app:app --reload --port 8000

Sin autenticacion: se asume alcanzable solo por el backend en la red local.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from bookmind_ml import __version__
from bookmind_ml.classifier import RUTA_METRICAS, clasificador
from bookmind_ml.dataset import DatasetVacioError, cargar_semilla
from bookmind_ml.readability import analizar
from bookmind_ml.taxonomy import MATERIA_LEGIBLE, NIVEL_LEGIBLE
from bookmind_ml.train import entrenar, guardar_metricas

# Mas corto que esto no es un libro; probablemente la extraccion de texto fallo.
MINIMO_CARACTERES = 200


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar, sin morir si no existe."""
    if clasificador.cargar():
        print(f"[ml-service] Modelo cargado (entrenado en {clasificador.entrenado_en}).")
    else:
        print(
            "[ml-service] AVISO: no hay modelo entrenado. /classify va a fallar "
            "hasta que ejecutes `python -m bookmind_ml.train` o llames a POST /train."
        )
    yield


app = FastAPI(
    title="BookMind AI — Enrutador Inteligente de Documentos",
    description="Clasifica libros escolares por materia (modelo entrenado) y "
                "nivel de lectura (indice Fernandez-Huerta).",
    version=__version__,
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Esquemas
# --------------------------------------------------------------------------

class PeticionClasificar(BaseModel):
    text: str = Field(..., description="Texto plano extraido del libro.")


class RespuestaSalud(BaseModel):
    status: str
    version: str
    modelo_cargado: bool
    entrenado_en: str | None
    modelo_demo: bool
    aviso: str | None = None


class RespuestaClasificar(BaseModel):
    materia: str
    materiaLegible: str
    nivel: str
    nivelLegible: str
    confidence: float
    bajaConfianza: bool
    featureImportance: list[dict[str, Any]]
    probabilidades: dict[str, float]
    legibilidad: dict[str, Any]
    # Clase a la que se refiere featureImportance. Difiere de `materia` cuando
    # la confianza es baja y la etiqueta se degrada a "otro".
    materiaExplicada: str
    # True si el modelo se entreno con datos sinteticos de prueba.
    modeloDemo: bool


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

AVISO_DEMO = (
    "El modelo cargado se entreno con textos sinteticos de prueba. Sirve para "
    "verificar el flujo de punta a punta, NO para reportar metricas. "
    "Reemplazalo con el corpus del CNB antes de la defensa."
)


@app.get("/health", response_model=RespuestaSalud)
def health() -> RespuestaSalud:
    return RespuestaSalud(
        status="ok",
        version=__version__,
        modelo_cargado=clasificador.esta_listo,
        entrenado_en=clasificador.entrenado_en,
        modelo_demo=clasificador.esta_listo and clasificador.es_demo,
        aviso=AVISO_DEMO if (clasificador.esta_listo and clasificador.es_demo) else None,
    )


@app.post("/classify", response_model=RespuestaClasificar)
def classify(peticion: PeticionClasificar) -> RespuestaClasificar:
    texto = peticion.text.strip()

    if len(texto) < MINIMO_CARACTERES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"El texto tiene {len(texto)} caracteres; se necesitan al menos "
                f"{MINIMO_CARACTERES}. Si el PDF es escaneado, la extraccion de "
                "texto fallo y hace falta OCR."
            ),
        )

    if not clasificador.esta_listo and not clasificador.cargar():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No hay modelo entrenado. Ejecuta POST /train primero.",
        )

    resultado = clasificador.clasificar(texto)

    return RespuestaClasificar(
        materia=resultado.materia,
        materiaLegible=MATERIA_LEGIBLE[resultado.materia],
        nivel=resultado.nivel,
        nivelLegible=NIVEL_LEGIBLE[resultado.nivel],
        confidence=resultado.confidence,
        bajaConfianza=resultado.baja_confianza,
        featureImportance=resultado.featureImportance,
        probabilidades=resultado.probabilidades,
        legibilidad=resultado.legibilidad,
        materiaExplicada=resultado.materia_explicada,
        modeloDemo=clasificador.es_demo,
    )


@app.post("/readability")
def readability(peticion: PeticionClasificar) -> dict[str, Any]:
    """Solo el indice de legibilidad, sin tocar el modelo."""
    return asdict(analizar(peticion.text))


@app.post("/train")
def train() -> dict[str, Any]:
    """Reentrena desde el corpus semilla. Sincrono: entrena en segundos."""
    try:
        ejemplos = cargar_semilla()
    except DatasetVacioError as error:
        raise HTTPException(
            status_code=status.HTTP_412_PRECONDITION_FAILED,
            detail=str(error),
        ) from error

    try:
        pipeline, resultado = entrenar(ejemplos)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_412_PRECONDITION_FAILED, detail=str(error)
        ) from error

    # Solo se publica si el entrenamiento termino: un fallo deja el anterior.
    clasificador.guardar(pipeline)
    guardar_metricas(resultado)

    return {
        "accuracy": resultado.accuracy,
        "accuracyBaseline": resultado.accuracy_baseline,
        "f1Macro": resultado.f1_macro,
        "nEntrenamiento": resultado.n_entrenamiento,
        "nTest": resultado.n_test,
        "distribucion": resultado.distribucion,
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """Ultimas metricas persistidas (evidencia de la seccion 4.5)."""
    if not RUTA_METRICAS.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Todavia no se ha entrenado el modelo.",
        )
    return json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))
