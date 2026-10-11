"""Microservicio de clasificacion; solo lo alcanza el backend en la red local."""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, Field

from bookmind_ml import __version__, estudio
from bookmind_ml.classifier import RUTA_METRICAS, clasificador
from bookmind_ml.ocr import configuracion as config_ocr
from bookmind_ml.ocr import rasterizado, reconocimiento
from bookmind_ml.ocr.trabajos import GestorTrabajos
from bookmind_ml.dataset import DatasetVacioError, cargar_semilla
from bookmind_ml.readability import analizar
from bookmind_ml.taxonomy import MATERIA_LEGIBLE, NIVEL_LEGIBLE
from bookmind_ml.train import entrenar, guardar_metricas

# Mas corto que esto no es un libro; probablemente la extraccion de texto fallo.
MINIMO_CARACTERES = 200

gestor_ocr = GestorTrabajos()


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
    gestor_ocr.limpiar_carpeta()
    tesseract = reconocimiento.estado_tesseract()
    if not tesseract.tiene(config_ocr.OCR_IDIOMA):
        print(
            "[ml-service] AVISO: Tesseract no esta instalado o le falta el idioma "
            f"'{config_ocr.OCR_IDIOMA}'. Los PDF escaneados no se podran digitalizar."
        )
    yield
    gestor_ocr.cerrar()


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
    tesseract_instalado: bool
    ocr_idioma_disponible: bool


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
    # Clase que describe featureImportance; con baja confianza `materia` es "otro".
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
    tesseract = reconocimiento.estado_tesseract()
    return RespuestaSalud(
        status="ok",
        version=__version__,
        modelo_cargado=clasificador.esta_listo,
        entrenado_en=clasificador.entrenado_en,
        modelo_demo=clasificador.esta_listo and clasificador.es_demo,
        aviso=AVISO_DEMO if (clasificador.esta_listo and clasificador.es_demo) else None,
        tesseract_instalado=tesseract.instalado,
        ocr_idioma_disponible=tesseract.tiene(config_ocr.OCR_IDIOMA),
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
    """Ultimas metricas persistidas del entrenamiento."""
    if not RUTA_METRICAS.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Todavia no se ha entrenado el modelo.",
        )
    return json.loads(RUTA_METRICAS.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Motor de estudio propio (sin LLM)
# --------------------------------------------------------------------------

class PaginaEntrada(BaseModel):
    pagina: int = Field(..., ge=1)
    texto: str
    # Solo en paginas digitalizadas con OCR; None es texto nativo del PDF.
    confianza: float | None = Field(default=None, ge=0, le=1)


class PeticionEstudio(BaseModel):
    paginas: list[PaginaEntrada] = Field(..., min_length=1)
    cantidad: int | None = Field(default=None, ge=1, le=40)
    # El umbral lo decide el backend; sin el, no se excluye nada.
    confianzaMinima: float | None = Field(default=None, ge=0, le=1)


# Un error de OCR en una pregunta o tarjeta se memoriza; en el resumen solo se lee.
TIPOS_QUE_EXIGEN_TEXTO_FIABLE = {"concepts", "flashcards", "quiz"}


# Generador por tipo de material; el backend usa las mismas claves.
GENERADORES = {
    "summary": (estudio.resumir, 8),
    "concepts": (estudio.glosario, 12),
    "flashcards": (estudio.generar_flashcards, 10),
    "quiz": (estudio.generar_quiz, 5),
    "timeline": (estudio.generar_linea_tiempo, 14),
}


@app.post("/study/{tipo}")
def study(tipo: str, peticion: PeticionEstudio) -> dict[str, Any]:
    """Genera un material desde el texto del libro; cada item trae su pagina de origen."""
    if tipo not in GENERADORES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No existe el generador '{tipo}'. Usa uno de: {', '.join(GENERADORES)}.",
        )

    generador, por_defecto = GENERADORES[tipo]
    fiables = peticion.paginas
    if tipo in TIPOS_QUE_EXIGEN_TEXTO_FIABLE and peticion.confianzaMinima is not None:
        minimo = peticion.confianzaMinima
        fiables = [p for p in peticion.paginas if p.confianza is None or p.confianza >= minimo]

    paginas = [{"pagina": p.pagina, "texto": p.texto} for p in fiables]
    items = generador(paginas, peticion.cantidad or por_defecto) if paginas else []

    return {
        "tipo": tipo,
        "origen": "motor",
        "items": items,
        "paginasExcluidas": len(peticion.paginas) - len(fiables),
    }


# --------------------------------------------------------------------------
# OCR de PDFs escaneados (trabajos asincronos)
# --------------------------------------------------------------------------

def _leer_lista_de_paginas(texto: str) -> list[int]:
    """Acepta '[1, 2, 5]' o '1,2,5': el backend manda JSON, curl a mano suele mandar lo segundo."""
    if not re.fullmatch(r"\s*\[?\s*\d+(\s*,\s*\d+)*\s*\]?\s*", texto):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="'paginas' debe ser una lista de numeros de pagina, por ejemplo [1, 2, 5].",
        )
    return sorted({int(numero) for numero in re.findall(r"\d+", texto)})


@app.post("/ocr/trabajos", status_code=status.HTTP_202_ACCEPTED)
async def crear_trabajo_ocr(
    archivo: UploadFile = File(..., description="PDF completo; se usa solo para las paginas pedidas."),
    paginas: str = Form(..., description="Paginas a digitalizar, 1-based: [1, 2, 5]."),
) -> dict[str, str]:
    limite = config_ocr.OCR_MAX_MB * 1024 * 1024
    if archivo.size is not None and archivo.size > limite:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"El PDF supera el maximo de {config_ocr.OCR_MAX_MB} MB.",
        )

    pdf = await archivo.read(limite + 1)
    if len(pdf) > limite:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"El PDF supera el maximo de {config_ocr.OCR_MAX_MB} MB.",
        )

    # Por contenido y no por nombre o mimetype: ambos los controla el cliente.
    if not pdf.startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="El archivo no es un PDF.",
        )

    numeros = _leer_lista_de_paginas(paginas)
    if not numeros or numeros[0] < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Las paginas empiezan en 1 y hay que pedir al menos una.",
        )
    if len(numeros) > config_ocr.OCR_MAX_PAGINAS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Se pidieron {len(numeros)} paginas; el maximo es {config_ocr.OCR_MAX_PAGINAS}.",
        )

    try:
        reconocimiento.exigir_tesseract(config_ocr.OCR_IDIOMA)
    except reconocimiento.OcrNoDisponibleError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
        ) from error

    try:
        total = rasterizado.contar_paginas(pdf)
    except rasterizado.PdfInvalidoError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)
        ) from error

    if numeros[-1] > total:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"El PDF tiene {total} paginas; se pidio la {numeros[-1]}.",
        )

    return {"id": gestor_ocr.crear(pdf, numeros)}


@app.get("/ocr/trabajos/{id_trabajo}")
def consultar_trabajo_ocr(id_trabajo: str) -> dict[str, Any]:
    """Las paginas solo vienen cuando el estado es 'listo'."""
    trabajo = gestor_ocr.consultar(id_trabajo)
    if trabajo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No existe ese trabajo de OCR; pudo vencer o el servicio se reinicio.",
        )
    return trabajo


@app.delete("/ocr/trabajos/{id_trabajo}", status_code=status.HTTP_204_NO_CONTENT)
def cancelar_trabajo_ocr(id_trabajo: str) -> Response:
    if not gestor_ocr.cancelar(id_trabajo):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No existe ese trabajo de OCR."
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)

