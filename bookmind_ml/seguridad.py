"""Token compartido con el backend: el ml-service no debe quedar abierto a cualquiera en la red."""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, Request, status

CABECERA = "X-ML-Token"
# /health queda abierto: es lo que consultan el backend y quien monitorea sin credenciales.
RUTAS_PUBLICAS = {"/health"}

TOKEN = os.environ.get("ML_SERVICE_TOKEN", "").strip()


def token_configurado() -> bool:
    return bool(TOKEN)


def verificar_token(
    request: Request,
    x_ml_token: str | None = Header(default=None, alias=CABECERA),
) -> None:
    # Sin token configurado no se exige nada: el desarrollo local sigue funcionando y el arranque lo avisa.
    if not TOKEN or request.url.path in RUTAS_PUBLICAS:
        return
    if not x_ml_token or not hmac.compare_digest(x_ml_token, TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Falta la cabecera {CABECERA} o no coincide con ML_SERVICE_TOKEN.",
        )
