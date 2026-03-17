"""Skills API router."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from adapters.skills.mcp_filesystem_adapter import (
    MCPFilesystemAdapter,
    SkillInstallError,
    SkillNotFoundError,
)
from db import get_db

router = APIRouter()


class SkillManifestResponse(BaseModel):
    """Skill manifest response."""

    name: str
    description: str
    version: str
    port: str | None = None
    author: str | None = None
    tags: list[str] = []
    requires: dict = {}
    config_schema: dict = {}
    capabilities: list[str] = []


class SkillResponse(BaseModel):
    """Full skill response."""

    manifest: SkillManifestResponse
    skill_path: str
    content: str
    is_active: bool


class SkillInstallRequest(BaseModel):
    """Skill installation request."""

    path_or_url: str


class SkillInvokeRequest(BaseModel):
    """Skill invocation request."""

    context: dict = {}


class SkillResultResponse(BaseModel):
    """Skill result response."""

    skill_name: str
    success: bool
    content: str | None = None
    error: str | None = None
    metadata: dict = {}


@router.get("/")
async def list_skills(
    db: AsyncSession = Depends(get_db),
) -> list[SkillManifestResponse]:
    """List all skills."""
    adapter = MCPFilesystemAdapter(db)
    manifests = await adapter.list_skills()

    return [
        SkillManifestResponse(
            name=m.name,
            description=m.description,
            version=m.version,
            port=m.port,
            author=m.author,
            tags=m.tags,
            requires=m.requires,
            config_schema=m.config_schema,
            capabilities=m.capabilities,
        )
        for m in manifests
    ]


@router.get("/{name}")
async def get_skill(
    name: str,
    db: AsyncSession = Depends(get_db),
) -> SkillResponse:
    """Get a skill by name."""
    adapter = MCPFilesystemAdapter(db)

    try:
        skill = await adapter.get_skill(name)

        return SkillResponse(
            manifest=SkillManifestResponse(
                name=skill.manifest.name,
                description=skill.manifest.description,
                version=skill.manifest.version,
                port=skill.manifest.port,
                author=skill.manifest.author,
                tags=skill.manifest.tags,
                requires=skill.manifest.requires,
                config_schema=skill.manifest.config_schema,
                capabilities=skill.manifest.capabilities,
            ),
            skill_path=skill.skill_path,
            content=skill.content,
            is_active=skill.is_active,
        )
    except SkillNotFoundError:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")


@router.post("/install")
async def install_skill(
    request: SkillInstallRequest,
    db: AsyncSession = Depends(get_db),
) -> SkillManifestResponse:
    """Install a skill from a path or URL."""
    adapter = MCPFilesystemAdapter(db)

    try:
        manifest = await adapter.install_skill(request.path_or_url)

        return SkillManifestResponse(
            name=manifest.name,
            description=manifest.description,
            version=manifest.version,
            port=manifest.port,
            author=manifest.author,
            tags=manifest.tags,
            requires=manifest.requires,
            config_schema=manifest.config_schema,
            capabilities=manifest.capabilities,
        )
    except SkillInstallError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{name}/invoke")
async def invoke_skill(
    name: str,
    request: SkillInvokeRequest,
    db: AsyncSession = Depends(get_db),
) -> SkillResultResponse:
    """Invoke a skill."""
    adapter = MCPFilesystemAdapter(db)

    result = await adapter.invoke_skill(name, request.context)

    return SkillResultResponse(
        skill_name=result.skill_name,
        success=result.success,
        content=result.content,
        error=result.error,
        metadata=result.metadata,
    )


@router.delete("/{name}")
async def uninstall_skill(
    name: str,
    db: AsyncSession = Depends(get_db),
):
    """Uninstall a skill."""
    adapter = MCPFilesystemAdapter(db)

    try:
        await adapter.uninstall_skill(name)
        return {"message": f"Skill '{name}' uninstalled"}
    except SkillNotFoundError:
        raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
    except SkillInstallError as e:
        raise HTTPException(status_code=400, detail=str(e))
