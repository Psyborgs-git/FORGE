"""FastAPI main application."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import hardware, inference, projects, skills
from api.websockets import inference_ws
from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("🔥 FORGE Backend starting...")
    print(f"📍 Database: {settings.DATABASE_URL}")
    print(f"🤖 Ollama: {settings.OLLAMA_BASE_URL}")

    # Initialize skills registry
    from adapters.skills.mcp_filesystem_adapter import MCPFilesystemAdapter
    from db import get_db

    async for db in get_db():
        skills_adapter = MCPFilesystemAdapter(db)
        await skills_adapter.scan_and_register_skills()
        print(f"✅ Skills scanned from {settings.FORGE_SKILLS_DIR}")
        break

    yield

    # Shutdown
    print("👋 FORGE Backend shutting down...")


app = FastAPI(
    title="FORGE API",
    description="Local AI Research & Build Studio",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(projects.router, prefix="/api", tags=["projects"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
app.include_router(hardware.router, prefix="/api/hardware", tags=["hardware"])
app.include_router(skills.router, prefix="/api/skills", tags=["skills"])
app.include_router(inference_ws.router, tags=["websockets"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "forge-backend"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "FORGE API",
        "version": "0.1.0",
        "docs": "/docs",
    }
