"""Trabajos de OCR en segundo plano; la interfaz es la que tendra la cola de la Fase 1.5."""

from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Tesseract abre hilos OpenMP propios; con varios trabajadores en paralelo eso solo compite por CPU.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

from . import configuracion, limpieza, preprocesado, rasterizado, reconocimiento  # noqa: E402

Estado = Literal["pendiente", "procesando", "listo", "error"]

Reconocedor = Callable[[Path, int], reconocimiento.PaginaReconocida]


def reconocer_pagina(ruta: Path, numero: int) -> reconocimiento.PaginaReconocida:
    imagen = rasterizado.rasterizar(ruta, numero, configuracion.OCR_DPI)
    binaria = preprocesado.preparar(imagen)
    crudo = reconocimiento.reconocer(binaria, configuracion.OCR_IDIOMA)
    return reconocimiento.PaginaReconocida(
        texto=limpieza.limpiar(crudo.texto), confianza=crudo.confianza
    )


@dataclass
class Trabajo:
    id: str
    paginas: list[int]
    ruta: Path | None
    estado: Estado = "pendiente"
    resultados: dict[int, reconocimiento.PaginaReconocida] = field(default_factory=dict)
    error: str | None = None
    terminado_en: float | None = None
    futuros: list[Future[None]] = field(default_factory=list)

    @property
    def procesadas(self) -> int:
        return len(self.resultados)


class GestorTrabajos:
    def __init__(
        self,
        reconocedor: Reconocedor = reconocer_pagina,
        trabajadores: int = configuracion.OCR_WORKERS,
        ttl_segundos: float = configuracion.OCR_TTL_SEGUNDOS,
        carpeta: Path = configuracion.CARPETA_TEMPORAL,
        reloj: Callable[[], float] = time.monotonic,
    ) -> None:
        self._reconocedor = reconocedor
        self._ejecutor = ThreadPoolExecutor(max_workers=trabajadores, thread_name_prefix="ocr")
        self._ttl = ttl_segundos
        self._carpeta = carpeta
        self._reloj = reloj
        self._trabajos: dict[str, Trabajo] = {}
        self._candado = threading.Lock()

    def limpiar_carpeta(self) -> None:
        """Al arrancar: un reinicio a mitad de trabajo deja PDFs huerfanos."""
        self._carpeta.mkdir(parents=True, exist_ok=True)
        for archivo in self._carpeta.glob("*.pdf"):
            archivo.unlink(missing_ok=True)

    def crear(self, pdf: bytes, paginas: list[int]) -> str:
        self.purgar()
        self._carpeta.mkdir(parents=True, exist_ok=True)

        descriptor, nombre = tempfile.mkstemp(suffix=".pdf", dir=self._carpeta)
        with os.fdopen(descriptor, "wb") as archivo:
            archivo.write(pdf)

        trabajo = Trabajo(id=uuid.uuid4().hex, paginas=sorted(set(paginas)), ruta=Path(nombre))

        with self._candado:
            self._trabajos[trabajo.id] = trabajo
            if not trabajo.paginas:
                sobrante = self._terminar(trabajo, "listo")
            else:
                sobrante = None
                trabajo.futuros = [
                    self._ejecutor.submit(self._procesar, trabajo, numero) for numero in trabajo.paginas
                ]

        self._borrar(sobrante)
        return trabajo.id

    def consultar(self, id_trabajo: str) -> dict[str, Any] | None:
        self.purgar()
        with self._candado:
            trabajo = self._trabajos.get(id_trabajo)
            return None if trabajo is None else self._instantanea(trabajo)

    def cancelar(self, id_trabajo: str) -> bool:
        with self._candado:
            trabajo = self._trabajos.pop(id_trabajo, None)
            if trabajo is None:
                return False
            sobrante = None
            if trabajo.terminado_en is None:
                sobrante = self._terminar(trabajo, "error", "Trabajo cancelado.")
        self._borrar(sobrante)
        return True

    def purgar(self) -> None:
        limite = self._reloj() - self._ttl
        with self._candado:
            vencidos = [
                id_trabajo
                for id_trabajo, trabajo in self._trabajos.items()
                if trabajo.terminado_en is not None and trabajo.terminado_en < limite
            ]
            for id_trabajo in vencidos:
                del self._trabajos[id_trabajo]

    def cerrar(self) -> None:
        self._ejecutor.shutdown(wait=False, cancel_futures=True)

    def _procesar(self, trabajo: Trabajo, numero: int) -> None:
        with self._candado:
            if trabajo.terminado_en is not None or trabajo.ruta is None:
                return
            trabajo.estado = "procesando"
            ruta = trabajo.ruta

        try:
            pagina = self._reconocedor(ruta, numero)
        except Exception as error:  # noqa: BLE001 - cualquier fallo de una pagina tumba el trabajo
            sobrante = None
            with self._candado:
                if trabajo.terminado_en is None:
                    sobrante = self._terminar(trabajo, "error", f"Fallo el OCR de la pagina {numero}: {error}")
            self._borrar(sobrante)
            return

        sobrante = None
        with self._candado:
            if trabajo.terminado_en is not None:
                return
            trabajo.resultados[numero] = pagina
            if trabajo.procesadas == len(trabajo.paginas):
                sobrante = self._terminar(trabajo, "listo")
        self._borrar(sobrante)

    def _terminar(self, trabajo: Trabajo, estado: Estado, error: str | None = None) -> Path | None:
        """Se llama con el candado tomado; devuelve el PDF a borrar una vez soltado."""
        trabajo.estado = estado
        trabajo.error = error
        trabajo.terminado_en = self._reloj()
        for futuro in trabajo.futuros:
            futuro.cancel()
        ruta, trabajo.ruta = trabajo.ruta, None
        return ruta

    @staticmethod
    def _borrar(ruta: Path | None) -> None:
        """Siempre: el PDF del estudiante no se queda en disco, termine como termine."""
        if ruta is None:
            return
        # En Windows no se borra un archivo abierto: se espera a que PDFium suelte el que esta rasterizando.
        with rasterizado.CANDADO_PDFIUM:
            ruta.unlink(missing_ok=True)

    @staticmethod
    def _instantanea(trabajo: Trabajo) -> dict[str, Any]:
        respuesta: dict[str, Any] = {
            "id": trabajo.id,
            "estado": trabajo.estado,
            "procesadas": trabajo.procesadas,
            "total": len(trabajo.paginas),
            "error": trabajo.error,
        }
        if trabajo.estado == "listo":
            respuesta["paginas"] = [
                {"numero": numero, "texto": pagina.texto, "confianza": pagina.confianza}
                for numero, pagina in sorted(trabajo.resultados.items())
            ]
        return respuesta
