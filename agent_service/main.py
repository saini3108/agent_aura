# Optional: Logging setup
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from agent_service.core.config import settings
from agent_service.core.flows.router import router as flow_runner_router
from agent_service.core.memory.redis_store import MemoryService
from agent_service.core.utils.logging_config import setup_logging
from agent_service.core.utils.middleware import RequestLoggingMiddleware

# ----------------------------------------
# 🧾 Logging Setup
# ----------------------------------------
setup_logging()
logger = logging.getLogger("aura_agent")

# Global memory service instance
memory_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global memory_service
    # Startup
    logger.info("Starting Agentic AI Banking Platform...")
    # Initialize memory service
    memory_service = MemoryService()
    await memory_service.initialize()

    # Store in app state
    app.state.memory = memory_service
    logger.info("Application started successfully")
    yield

    # Shutdown
    logger.info("Shutting down application...")
    if memory_service:
        await memory_service.close()
    logger.info("Application shutdown complete")


# ----------------------------------------
# 🚀 App Factory
# ----------------------------------------
app = FastAPI(
    title="AURA Agentic AI Service",
    description="Agent orchestration layer for risk automation workflows.",
    version="0.1.0",
    lifespan=lifespan,
)

# ----------------------------------------
# 🛡️ Security: CORS Middleware
# ----------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# ----------------------------------------
# 📜 Request Logging Middleware
# ----------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("📥 %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled exception during request")
        raise
    logger.info("📤 %s %s → %s", request.method, request.url.path, response.status_code)
    return response


# ----------------------------------------
# 🏥 Health & Root Routes
# ----------------------------------------
@app.get("/", tags=["health"])
def root():
    """Root endpoint with service metadata."""
    return {
        "message": "Welcome to AURA Agentic AI Server 🌐",
        "status": "OK",
        "version": app.version,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check memory service connection
        if hasattr(app.state, 'memory') and app.state.memory:
            await app.state.memory.health_check()

        return {
            "status": "healthy",
            "services": {
                "memory": "connected",
                "api": "running"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

# ----------------------------------------
# 🧯 Global Exception Handler
# ----------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("🔥 Unhandled exception occurred")
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal Server Error"},
    )


# ----------------------------------------
# 📦 Auto-load Future Routers
# ----------------------------------------
app.include_router(flow_runner_router, prefix="/api/v1")


# ----------------------------------------
# 🚀 Dev Entry (not for prod gunicorn/uvicorn workers)
# ----------------------------------------
if __name__ == "__main__":
    logger.warning("🚧 Starting development server with reload enabled.")
    uvicorn.run(
        "agent_service.main:app",
        host="127.0.0.1",
        port=8000,
        reload=settings.DEBUG,  # 🔁 Only reload in dev
        workers=1 if settings.DEBUG else 4,
        timeout_keep_alive=60,
        log_level="info",
    )
