"""
FastAPI-based API server for URCM.
Replaces the single-threaded HTTPServer with async support.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field

from urcm.config import get_settings

logger = logging.getLogger(__name__)


# ---------- Configuration ----------

settings = get_settings()

# Security scheme for OpenAPI
security = HTTPBearer(auto_error=False)


# ---------- Auth Dependency ----------

def verify_token(authorization: Optional[str] = Header(None)) -> None:
    import hmac
    if not settings.metrics_token:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization scheme")
    if not hmac.compare_digest(authorization[len("Bearer "):], settings.metrics_token):
        raise HTTPException(status_code=403, detail="Invalid token")


# ---------- Request/Response Models ----------

class ReasonRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Input text for reasoning",
        examples=["All humans are mortal. Socrates is human."]
    )
    max_steps: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum reasoning steps",
        examples=[50]
    )
    beam_width: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Beam width for search",
        examples=[3]
    )


class ReasonResponse(BaseModel):
    text: str = Field(description="Original input text")
    final_mu: float = Field(description="Final convergence value (0-1)", examples=[0.87])
    steps: int = Field(description="Number of reasoning steps taken", examples=[12])
    converged: bool = Field(description="Whether reasoning converged", examples=[True])


class HealthResponse(BaseModel):
    ok: bool = Field(description="Overall system health")
    checks: dict = Field(description="Individual health checks")
    env: str = Field(description="Environment name")
    timestamp: float = Field(description="Unix timestamp")


class ValidateResponse(BaseModel):
    status: str = Field(description="System status: ok or degraded")
    checks: dict = Field(description="Individual health checks")


class ErrorResponse(BaseModel):
    detail: str = Field(description="Error message")


# ---------- App Setup ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"URCM API starting on {settings.metrics_bind}:{settings.metrics_port}")
    yield
    logger.info("URCM API shutting down")


app = FastAPI(
    title="URCM API",
    description=(
        "Unified μ-Resonance Cognitive Mesh (URCM) API\n\n"
        "A frequency-based reasoning system that replaces token-based processing "
        "with continuous frequency-based representations.\n\n"
        "**Features:**\n"
        "- 100% local & private (no cloud dependency)\n"
        "- Deterministic reasoning (μ-stability for reproducible outputs)\n"
        "- Value-grounded intelligence (built-in ethics)\n"
        "- Memory efficient (<45MB footprint)\n"
        "- O(1) attractor lookup\n"
        "- Self-healing metacognition\n\n"
        "**Authentication:** Bearer token required for all endpoints "
        "(set URCM_METRICS_TOKEN environment variable)."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Health", "description": "Health and monitoring endpoints"},
        {"name": "Reasoning", "description": "Core reasoning endpoints"},
    ],
)

# Add Bearer token security scheme to OpenAPI
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Bearer token authentication. Set URCM_METRICS_TOKEN env var."
        }
    }
    # Apply security to all paths
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
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

@app.get(
    "/health",
    response_model=HealthResponse,
    dependencies=[Depends(verify_token)],
    tags=["Health"],
    summary="System health check",
    description="Returns overall system health and individual component checks.",
    responses={
        200: {"description": "Health check successful"},
        401: {"model": ErrorResponse, "description": "Missing or invalid authorization"},
        403: {"model": ErrorResponse, "description": "Invalid token"},
    }
)
async def health(env: Optional[str] = None):
    system = get_system()
    checks = system.validate_system()
    return {
        "ok": checks.get("overall_health", False),
        "checks": checks,
        "env": env or settings.env,
        "timestamp": time.time(),
    }


@app.get(
    "/metrics",
    response_class=PlainTextResponse,
    dependencies=[Depends(verify_token)],
    tags=["Health"],
    summary="Prometheus metrics",
    description="Returns Prometheus-formatted metrics for monitoring.",
    responses={
        200: {"description": "Metrics in Prometheus text format", "content": {"text/plain": {}}},
        401: {"model": ErrorResponse, "description": "Missing or invalid authorization"},
        403: {"model": ErrorResponse, "description": "Invalid token"},
    }
)
async def metrics():
    from urcm.ops.metrics_exporter import render_metrics
    return render_metrics(os.getcwd(), env=settings.env or None)


@app.post(
    "/api/reason",
    response_model=ReasonResponse,
    dependencies=[Depends(verify_token)],
    tags=["Reasoning"],
    summary="Process reasoning query",
    description=(
        "Process a natural language query through the URCM resonance engine. "
        "Returns the reasoning result with convergence metrics."
    ),
    responses={
        200: {"description": "Reasoning completed successfully"},
        401: {"model": ErrorResponse, "description": "Missing or invalid authorization"},
        403: {"model": ErrorResponse, "description": "Invalid token"},
        500: {"model": ErrorResponse, "description": "Reasoning failed"},
    }
)
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


@app.get(
    "/api/validate",
    response_model=ValidateResponse,
    dependencies=[Depends(verify_token)],
    tags=["Health"],
    summary="Full system validation",
    description="Returns detailed validation of all system components.",
    responses={
        200: {"description": "Validation completed"},
        401: {"model": ErrorResponse, "description": "Missing or invalid authorization"},
        403: {"model": ErrorResponse, "description": "Invalid token"},
    }
)
async def validate():
    system = get_system()
    checks = system.validate_system()
    return {"status": "ok" if checks.get("overall_health") else "degraded", "checks": checks}


# ---------- Entrypoint ----------

def main():
    import uvicorn
    uvicorn.run(
        "urcm.api:app",
        host=settings.metrics_bind,
        port=settings.metrics_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
