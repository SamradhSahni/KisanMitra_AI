import os
import sys
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, ".")

from backend.api.routes import chat, msp, feedback
from backend.api.models.schemas import HealthResponse

APP_VERSION  = "1.0.0"
APP_START    = time.time()


# ── Lifespan: load pipeline once at startup ───────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Loads the RAG pipeline once on startup,
    cleans up on shutdown.
    """
    logger.info("=" * 60)
    logger.info("KisanMitra AI — Starting up...")
    logger.info("=" * 60)

    # Load full RAG pipeline (model + retriever + warmup)
    from backend.rag.pipeline import KisanMitraRAGPipeline
    pipeline = KisanMitraRAGPipeline()
    pipeline.load()

    # Store in app state so routes can access it
    app.state.pipeline = pipeline

    logger.success("Startup complete — API ready to serve requests")
    logger.info("=" * 60)

    yield    # server runs here

    # Shutdown
    logger.info("Shutting down KisanMitra AI...")
    if hasattr(pipeline, 'inference_engine') and pipeline.inference_engine:
        pipeline.inference_engine.unload()
    logger.info("Shutdown complete")


# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(
    title="KisanMitra AI API",
    description=(
        "Hindi-first agricultural advisory chatbot API for Indian farmers. "
        "Powered by fine-tuned mT5-base with QLoRA and RAG pipeline."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── CORS ──────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",     # React dev server
        "http://localhost:5173",     # Vite dev server
        "https://*.amplifyapp.com",  # AWS Amplify (fill in Task 22)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start  = time.time()
    method = request.method
    path   = request.url.path

    response = await call_next(request)

    duration_ms = int((time.time() - start) * 1000)
    logger.info(f"{method} {path} → {response.status_code} ({duration_ms}ms)")
    return response


# ── Global exception handler ──────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


# ── Health endpoint ───────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Health"],
)
async def health(request: Request):
    pipeline   = request.app.state.pipeline
    model_ok   = pipeline.is_ready()
    rag_ok     = pipeline.retriever.is_ready() if pipeline.retriever else False
    uptime     = round(time.time() - APP_START, 1)

    return HealthResponse(
        status  = "ok" if model_ok else "degraded",
        model   = "ready" if model_ok else "not_ready",
        rag     = "ready" if rag_ok  else "not_ready",
        version = APP_VERSION,
        uptime_s= uptime,
    )


# ── Register routers ──────────────────────────────────────────────────
app.include_router(
    chat.router,
    prefix="/api/v1",
    tags=["Chat"],
)
app.include_router(
    msp.router,
    prefix="/api/v1",
    tags=["MSP Prices"],
)
app.include_router(
    feedback.router,
    prefix="/api/v1",
    tags=["Feedback"],
)


# ── Root ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "🌾 KisanMitra AI API",
        "docs":    "/docs",
        "health":  "/health",
        "version": APP_VERSION,
    }