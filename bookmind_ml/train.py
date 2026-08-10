"""Entrenamiento y evaluacion del clasificador (secciones 4.2 y 4.5).

    python -m bookmind_ml.train

Produce `models/clasificador_materia.joblib` y `models/metricas.json` con
accuracy, precision/recall por clase, matriz de confusion y baseline.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .classifier import RUTA_METRICAS, clasificador, construir_pipeline
from .dataset import EjemploEntrenamiento, cargar_semilla, resumen_por_clase

# 25%: con un corpus pequeno deja test interpretable sin quedarse sin train.
PROPORCION_TEST = 0.25
SEMILLA_ALEATORIA = 42


@dataclass
class ResultadoEntrenamiento:
    accuracy: float
    accuracy_baseline: float
    f1_macro: float
    reporte_por_clase: dict
    matriz_confusion: list[list[int]]
    etiquetas: list[str]
    n_entrenamiento: int
    n_test: int
    distribucion: dict[str, int]
    cv_accuracy_media: float | None
    cv_accuracy_desviacion: float | None


def _elegir_min_df(n_documentos: int) -> int:
    """Decide `min_df` una sola vez, a partir del corpus completo.

    Con corpus chico `min_df=2` puede vaciar el vocabulario TF-IDF entero, y
    usar valores distintos en evaluacion y despliegue da metricas de un modelo
    que no es el que se sirve.
    """
    if n_documentos < 20:
        print(
            f"[train] Corpus de {n_documentos} documentos: min_df=1 "
            "para no vaciar el vocabulario TF-IDF."
        )
        return 1
    return 2


def entrenar(ejemplos: list[EjemploEntrenamiento]) -> tuple[Pipeline, ResultadoEntrenamiento]:
    """Entrena y evalua. Devuelve `(pipeline, metricas)` sin escribir en disco."""
    X = np.asarray([[e.texto] for e in ejemplos], dtype=object)
    y = np.asarray([e.materia for e in ejemplos])

    distribucion = resumen_por_clase(ejemplos)
    conteos = Counter(y)
    minimo_por_clase = min(conteos.values())

    print(f"[train] {len(ejemplos)} documentos, {len(conteos)} clases presentes.")
    for materia, n in sorted(distribucion.items(), key=lambda kv: -kv[1]):
        print(f"        {materia:24s} {n:4d}")

    if len(conteos) < 2:
        raise ValueError(
            "El corpus tiene una sola materia. Un clasificador necesita al "
            "menos dos clases para que las metricas signifiquen algo."
        )

    # `stratify` exige al menos un ejemplo de cada clase en cada lado.
    puede_estratificar = minimo_por_clase >= 2
    if not puede_estratificar:
        clases_escasas = [c for c, n in conteos.items() if n < 2]
        print(
            f"[train] AVISO: {clases_escasas} tienen 1 solo ejemplo. Se hace el "
            "split sin estratificar y las metricas por clase seran fragiles. "
            "Agrega mas documentos del CNB a esas materias."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=PROPORCION_TEST,
        random_state=SEMILLA_ALEATORIA,
        stratify=y if puede_estratificar else None,
    )

    # Un solo valor para los tres pipelines de aqui abajo. Ver `_elegir_min_df`.
    min_df = _elegir_min_df(len(X))

    pipeline = construir_pipeline(min_df=min_df)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # --- Baseline obligatorio: predecir siempre la clase mayoritaria ---
    baseline = DummyClassifier(strategy="most_frequent", random_state=SEMILLA_ALEATORIA)
    baseline.fit(X_train, y_train)
    accuracy_baseline = float(accuracy_score(y_test, baseline.predict(X_test)))

    accuracy = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    etiquetas = sorted(set(y))
    matriz = confusion_matrix(y_test, y_pred, labels=etiquetas).tolist()
    reporte = classification_report(
        y_test, y_pred, labels=etiquetas, zero_division=0, output_dict=True
    )

    # --- Validacion cruzada: con corpus chico, un solo split enganza ---
    cv_media = cv_desviacion = None
    if puede_estratificar and minimo_por_clase >= 3:
        n_splits = min(5, minimo_por_clase)
        puntajes = cross_val_score(
            construir_pipeline(min_df=min_df), X, y,
            cv=StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=SEMILLA_ALEATORIA
            ),
            scoring="accuracy",
        )
        cv_media = float(puntajes.mean())
        cv_desviacion = float(puntajes.std())
        print(
            f"[train] Validacion cruzada ({n_splits} folds): "
            f"{cv_media:.3f} +/- {cv_desviacion:.3f}"
        )
    else:
        print(
            "[train] Sin validacion cruzada: hacen falta >=3 ejemplos por clase."
        )

    # El modelo final usa todo el corpus; el split ya dio su estimacion.
    pipeline_final = construir_pipeline(min_df=min_df)
    pipeline_final.fit(X, y)

    metricas = ResultadoEntrenamiento(
        accuracy=accuracy,
        accuracy_baseline=accuracy_baseline,
        f1_macro=f1_macro,
        reporte_por_clase=reporte,
        matriz_confusion=matriz,
        etiquetas=etiquetas,
        n_entrenamiento=len(X_train),
        n_test=len(X_test),
        distribucion=distribucion,
        cv_accuracy_media=cv_media,
        cv_accuracy_desviacion=cv_desviacion,
    )

    return pipeline_final, metricas


def guardar_metricas(resultado: ResultadoEntrenamiento) -> None:
    RUTA_METRICAS.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "accuracy": resultado.accuracy,
        "accuracy_baseline_clase_mayoritaria": resultado.accuracy_baseline,
        "mejora_sobre_baseline": round(
            resultado.accuracy - resultado.accuracy_baseline, 4
        ),
        "f1_macro": resultado.f1_macro,
        "cv_accuracy_media": resultado.cv_accuracy_media,
        "cv_accuracy_desviacion": resultado.cv_accuracy_desviacion,
        "n_entrenamiento": resultado.n_entrenamiento,
        "n_test": resultado.n_test,
        "distribucion_por_clase": resultado.distribucion,
        "etiquetas": resultado.etiquetas,
        "matriz_confusion": resultado.matriz_confusion,
        "reporte_por_clase": resultado.reporte_por_clase,
    }
    RUTA_METRICAS.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _imprimir_resumen(resultado: ResultadoEntrenamiento) -> None:
    print("\n" + "=" * 62)
    print("  EVALUACION DEL CLASIFICADOR DE MATERIA")
    print("=" * 62)
    print(f"  Accuracy (test)            : {resultado.accuracy:.3f}")
    print(f"  Baseline (clase mayoritaria): {resultado.accuracy_baseline:.3f}")
    print(f"  F1 macro                   : {resultado.f1_macro:.3f}")

    delta = resultado.accuracy - resultado.accuracy_baseline
    veredicto = "SUPERA el baseline" if delta > 0 else "NO supera el baseline"
    print(f"  -> {veredicto} por {delta:+.3f}")

    print("\n  Matriz de confusion (filas = real, columnas = predicho):")
    ancho = max(len(e) for e in resultado.etiquetas) + 2
    print(" " * (ancho + 2) + "".join(f"{e[:6]:>8s}" for e in resultado.etiquetas))
    for etiqueta, fila in zip(resultado.etiquetas, resultado.matriz_confusion):
        print(f"  {etiqueta:<{ancho}s}" + "".join(f"{v:8d}" for v in fila))
    print("=" * 62 + "\n")


def main() -> int:
    ejemplos = cargar_semilla()
    pipeline, resultado = entrenar(ejemplos)
    clasificador.guardar(pipeline)
    guardar_metricas(resultado)
    _imprimir_resumen(resultado)
    print(f"Modelo guardado. Metricas en {RUTA_METRICAS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
