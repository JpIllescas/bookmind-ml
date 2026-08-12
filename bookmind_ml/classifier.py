"""Pieza 3: TF-IDF + features numericas -> LogisticRegression -> materia."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, StandardScaler

from .features import NOMBRES_FEATURES, FeaturesNumericas
from .readability import analizar
from .taxonomy import MATERIAS

RUTA_MODELOS = Path(__file__).resolve().parent.parent / "models"
RUTA_MODELO = RUTA_MODELOS / "clasificador_materia.joblib"
RUTA_METRICAS = RUTA_MODELOS / "metricas.json"

# Umbral por debajo del cual no se fuerza una clase.
UMBRAL_CONFIANZA = 0.45


@dataclass(frozen=True)
class ResultadoClasificacion:
    """Respuesta de /classify."""

    materia: str
    nivel: str
    confidence: float
    featureImportance: list[dict[str, float | str]]
    probabilidades: dict[str, float]
    legibilidad: dict[str, float | str | int]
    baja_confianza: bool
    # Clase que describe `featureImportance`; con baja confianza `materia` es "otro".
    materia_explicada: str


# Funciones de modulo y no lambdas: pickle las resuelve por nombre.

def _aplanar(X):
    """Convierte la columna (n, 1) en la secuencia 1-D que espera TfidfVectorizer."""
    return np.asarray(X).ravel()


def _recortar(M):
    """Acota cada feature escalada a +/-5 sigma."""
    return np.clip(M, -5.0, 5.0)


def construir_pipeline(min_df: int = 2) -> Pipeline:
    """Arma el pipeline sin entrenar: ambas ramas reciben el mismo texto crudo."""
    aplanar = FunctionTransformer(
        _aplanar,
        validate=False,
        feature_names_out="one-to-one",
    )

    rama_tfidf = Pipeline([
        ("aplanar", aplanar),
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            # Unigramas y bigramas: "numeros naturales", "guerra civil".
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=0.85,
            max_features=30_000,
            sublinear_tf=True,
        )),
    ])

    rama_numerica = Pipeline([
        ("aplanar", aplanar),
        ("features", FeaturesNumericas()),
        ("escalar", StandardScaler()),
        # Limita cuanto puede pesar una feature fuera de distribucion.
        ("recortar", FunctionTransformer(
            _recortar,
            validate=False,
            feature_names_out="one-to-one",
        )),
        # A norma 1, igual que TF-IDF, para que ambos bloques pesen lo mismo.
        ("normalizar", Normalizer()),
    ])

    combinador = ColumnTransformer(
        transformers=[
            ("tfidf", rama_tfidf, [0]),
            ("numericas", rama_numerica, [0]),
        ],
    )

    return Pipeline([
        ("features", combinador),
        ("clf", LogisticRegression(
            max_iter=2000,
            # El corpus del CNB no estara balanceado entre areas.
            class_weight="balanced",
            C=1.0,
            random_state=42,
        )),
    ])


def _nombres_features(pipeline: Pipeline) -> list[str]:
    """Devuelve los nombres de todas las columnas que ve el clasificador."""
    combinador: ColumnTransformer = pipeline.named_steps["features"]
    tfidf: TfidfVectorizer = (
        combinador.named_transformers_["tfidf"].named_steps["tfidf"]
    )
    nombres_tfidf = [f"palabra:{t}" for t in tfidf.get_feature_names_out()]
    nombres_num = [f"estructura:{n}" for n in NOMBRES_FEATURES]
    return nombres_tfidf + nombres_num


class ClasificadorMateria:
    """Envoltorio con carga perezosa del modelo entrenado."""

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._nombres: list[str] | None = None
        self._entrenado_en: str | None = None
        self._es_demo: bool = False

    # --- Ciclo de vida ---

    @property
    def esta_listo(self) -> bool:
        return self._pipeline is not None

    @property
    def es_demo(self) -> bool:
        """True si el modelo se entreno con datos sinteticos."""
        return self._es_demo

    def cargar(self) -> bool:
        """Carga el modelo del disco. Devuelve False si no hay ninguno."""
        if not RUTA_MODELO.exists():
            return False
        paquete = joblib.load(RUTA_MODELO)
        self._pipeline = paquete["pipeline"]
        self._nombres = paquete["nombres_features"]
        self._entrenado_en = paquete.get("entrenado_en")
        # Por defecto True: equivocarse hacia "es demo" es el error barato.
        self._es_demo = bool(paquete.get("es_demo", True))
        return True

    def guardar(self, pipeline: Pipeline, es_demo: bool = False) -> None:
        RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
        entrenado_en = datetime.now(timezone.utc).isoformat()
        joblib.dump(
            {
                "pipeline": pipeline,
                "nombres_features": _nombres_features(pipeline),
                "entrenado_en": entrenado_en,
                "es_demo": es_demo,
                "clases": list(pipeline.named_steps["clf"].classes_),
            },
            RUTA_MODELO,
            compress=3,
        )
        self._pipeline = pipeline
        self._nombres = _nombres_features(pipeline)
        self._entrenado_en = entrenado_en
        self._es_demo = es_demo

    @property
    def entrenado_en(self) -> str | None:
        return self._entrenado_en

    # --- Inferencia ---

    def clasificar(self, texto: str, top_n: int = 8) -> ResultadoClasificacion:
        """Predice materia + nivel y explica la decision."""
        if self._pipeline is None:
            raise RuntimeError(
                "El clasificador no esta entrenado. Ejecuta `python -m "
                "bookmind_ml.train` antes de llamar a /classify."
            )

        entrada = np.asarray([[texto]], dtype=object)
        probabilidades = self._pipeline.predict_proba(entrada)[0]
        clases = self._pipeline.named_steps["clf"].classes_

        indice_ganador = int(np.argmax(probabilidades))
        confianza = float(probabilidades[indice_ganador])
        materia = str(clases[indice_ganador])

        # Con poca confianza no se fuerza una clase.
        baja_confianza = confianza < UMBRAL_CONFIANZA
        if baja_confianza:
            materia = "otro"

        legibilidad = analizar(texto)

        return ResultadoClasificacion(
            materia=materia,
            materia_explicada=str(clases[indice_ganador]),
            nivel=legibilidad.nivel,
            confidence=round(confianza, 4),
            featureImportance=self._explicar(entrada, indice_ganador, top_n),
            probabilidades={
                str(clase): round(float(p), 4)
                for clase, p in zip(clases, probabilidades)
            },
            legibilidad=asdict(legibilidad),
            baja_confianza=baja_confianza,
        )

    def _explicar(
        self, entrada: np.ndarray, indice_clase: int, top_n: int
    ) -> list[dict[str, float | str]]:
        """Top-N features que mas empujaron hacia la clase ganadora."""
        assert self._pipeline is not None and self._nombres is not None

        combinador: ColumnTransformer = self._pipeline.named_steps["features"]
        clasificador: LogisticRegression = self._pipeline.named_steps["clf"]

        vector = combinador.transform(entrada)
        # La rama TF-IDF devuelve matriz dispersa; la union tambien.
        vector_denso = np.asarray(
            vector.todense() if hasattr(vector, "todense") else vector
        ).ravel()

        coeficientes = clasificador.coef_
        # Con solo dos clases, sklearn guarda una unica fila de coeficientes.
        if coeficientes.shape[0] == 1:
            fila = coeficientes[0] if indice_clase == 1 else -coeficientes[0]
        else:
            fila = coeficientes[indice_clase]

        contribuciones = vector_denso * fila
        # Solo interesan las que empujan A FAVOR de la clase elegida.
        indices = np.argsort(contribuciones)[::-1][:top_n]

        return [
            {
                "feature": self._nombres[i],
                "contribucion": round(float(contribuciones[i]), 4),
                "valor": round(float(vector_denso[i]), 4),
            }
            for i in indices
            if contribuciones[i] > 0
        ]


# Instancia unica que usa la app de FastAPI.
clasificador = ClasificadorMateria()

__all__ = [
    "ClasificadorMateria",
    "ResultadoClasificacion",
    "clasificador",
    "construir_pipeline",
    "MATERIAS",
    "RUTA_METRICAS",
    "RUTA_MODELO",
]
