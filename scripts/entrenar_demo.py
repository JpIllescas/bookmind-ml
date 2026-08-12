"""Entrena un modelo de juguete con textos sinteticos, solo para probar el flujo."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from bookmind_ml.classifier import (  # noqa: E402
    RUTA_METRICAS,
    RUTA_MODELO,
    clasificador,
    construir_pipeline,
)
from bookmind_ml.demo_corpus import construir_corpus_demo  # noqa: E402


def main() -> int:
    print("=" * 68)
    print("  MODELO DE PRUEBA — datos sinteticos, NO usar para metricas")
    print("=" * 68)

    # Si hay metricas.json ya hubo entrenamiento real y no se debe pisar.
    if RUTA_METRICAS.exists() and "--force" not in sys.argv:
        print(
            f"\n  ABORTADO: existe {RUTA_METRICAS.name}, o sea que ya se entreno\n"
            "  con el corpus real. Este script reemplazaria ese modelo por uno\n"
            "  de juguete y las metricas guardadas dejarian de corresponderle.\n\n"
            "  Si de verdad quieres volver al modelo de prueba:\n"
            "      python scripts/entrenar_demo.py --force\n"
            "  Para regenerar el modelo real:\n"
            "      python -m bookmind_ml.train\n",
            file=sys.stderr,
        )
        return 1

    corpus = construir_corpus_demo()
    X = np.asarray([[texto] for texto, _ in corpus], dtype=object)
    y = np.asarray([materia for _, materia in corpus])

    print(f"\n  {len(corpus)} documentos sinteticos, {len(set(y))} materias.")

    # min_df=1: el corpus es diminuto y con 2 se vaciaria el vocabulario.
    pipeline = construir_pipeline(min_df=1)
    pipeline.fit(X, y)

    clasificador.guardar(pipeline, es_demo=True)

    print(f"  Modelo guardado en {RUTA_MODELO}")

    print(
        "\n  Ya puedes arrancar el servicio y probar /classify.\n"
        "  /health y cada respuesta van a indicar que el modelo es de prueba.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
