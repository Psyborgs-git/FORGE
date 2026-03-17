"""FastAPI dependency injection for ports and adapters."""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from adapters.inference.ollama_adapter import OllamaAdapter
from adapters.skills.mcp_filesystem_adapter import MCPFilesystemAdapter
from core.ports import InferencePort, SkillRegistryPort
from db import get_db
from services.hardware_advisor import hardware_advisor, HardwareAdvisor


async def get_inference_port() -> AsyncGenerator[InferencePort, None]:
    """Dependency for getting inference port implementation."""
    adapter = OllamaAdapter()
    try:
        yield adapter
    finally:
        await adapter.close()


async def get_skills_port(
    db: AsyncSession = None,
) -> AsyncGenerator[SkillRegistryPort, None]:
    """Dependency for getting skills registry port implementation."""
    if db is None:
        async for session in get_db():
            db = session
            break
    adapter = MCPFilesystemAdapter(db)
    yield adapter


def get_hardware_advisor() -> HardwareAdvisor:
    """Dependency for getting hardware advisor."""
    return hardware_advisor
