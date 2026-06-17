"""
FastAPI-based API server for URCM.
Replaces the single-threaded HTTPServer with async support.
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------- Env Configuration ----------

METRICS_TOKEN = os.environ.get("URCM_METRICS_TOKEN", "")
METRICS_PORT = int(os.environ.get("URCM_METRICS_PORT", "8008"))
METRICS_BIND = os.environ.get("URCM_METRICS_BIND", "127.0.0.1")
ENV_NAME = os.environ.get("URCM_ENV", "")
SLO_MIN_FINAL_MU = float(os.environ.get("URCM_SLO_MIN_FINAL_MU", "0.4"))
SLO_MAX_STEPS_RATE = float(os.environ.get("URCM_SLO_MAX_STEPS_RATE", "0.5"))
CORS_ORIGINS = os.environ.get("URCM_CORS_ORIGINS", "*").split(",")
RATE_LIMIT = int(os.environ.get("URCM_RATE_LIMIT", "100"))


# ---------- Auth Dependency ----------

def verify_token(authorization: Optional[str] = Header(None)) -> None:
    if not METRICS_TOKEN:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization scheme")
    if authorization[len("Bearer "):] != METRICS_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


# ---------- Request/Response Models ----------

class ReasonRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)
    max_steps: int = Field(default=50, ge=1, le=500)
    beam_width: int = Field(default=3, ge=1, le=10)


class ReasonResponse(BaseModel):
    text: str
    final_mu: float
    steps: int
    converged: bool


# ---------- App Setup ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"URCM API starting on {METRICS_BIND}:{METRICS_PORT}")
    yield
    logger.info("URCM API shutting down")


app = FastAPI(
    title="URCM API",
    description="Unified μ-Resonance Cognitive Mesh API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Lazy System Initialization ----------

_system = None

def get_system():
    global _system
    if _system is None:
        from urcm.core.system import URCMSystem
        _system = URCMSystem()
    return _system


# ---------- Routes ----------

@app.get("/health", dependencies=[Depends(verify_token)])
async def health(env: Optional[str] = None):
    system = get_system()
    checks = system.validate_system()
    return {
        "ok": checks.get("overall_health", False),
        "checks": checks,
        "env": env or ENV_NAME,
        "timestamp": time.time(),
    }


@app.get("/metrics", response_class=PlainTextResponse, dependencies=[Depends(verify_token)])
async def metrics():
    from urcm.ops.metrics_exporter import render_metrics
    return render_metrics(os.getcwd(), env=ENV_NAME or None)


@app.post("/api/reason", dependencies=[Depends(verify_token)])
async def reason(req: ReasonRequest):
    system = get_system()
    try:
        result = system.process_query(req.text)
        final_mu = float(result.mu_trajectory[-1]) if result.mu_trajectory else 0.0
        converged = len(result.mu_trajectory) >= 2 and abs(result.mu_trajectory[-1] - result.mu_trajectory[-2]) < 1e-3
        return ReasonResponse(
            text=req.text,
            final_mu=final_mu,
            steps=len(result.mu_trajectory),
            converged=converged,
        )
    except Exception as e:
        logger.exception("Reasoning failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/validate", dependencies=[Depends(verify_token)])
async def validate():
    system = get_system()
    checks = system.validate_system()
    return {"status": "ok" if checks.get("overall_health") else "degraded", "checks": checks}


# ---------- Entrypoint ----------

def main():
    import uvicorn
    uvicorn.run(
        "urcm.api:app",
        host=METRICS_BIND,
        port=METRICS_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
