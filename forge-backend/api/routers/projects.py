"""Projects API router."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import get_db
from db.models import Project

router = APIRouter()


class ProjectCreate(BaseModel):
    """Project creation request."""

    name: str
    mode: str = "guided"
    metadata: dict = {}


class ProjectUpdate(BaseModel):
    """Project update request."""

    name: str | None = None
    mode: str | None = None
    metadata: dict | None = None


class ProjectResponse(BaseModel):
    """Project response."""

    id: UUID
    name: str
    mode: str
    created_at: str
    metadata: dict

    model_config = {"from_attributes": True}


@router.get("/projects")
async def list_projects(db: AsyncSession = Depends(get_db)) -> list[ProjectResponse]:
    """List all projects."""
    result = await db.execute(select(Project).order_by(Project.created_at.desc()))
    projects = result.scalars().all()
    return [
        ProjectResponse(
            id=p.id,
            name=p.name,
            mode=p.mode,
            created_at=p.created_at.isoformat(),
            metadata=p.metadata,
        )
        for p in projects
    ]


@router.post("/projects")
async def create_project(
    project: ProjectCreate, db: AsyncSession = Depends(get_db)
) -> ProjectResponse:
    """Create a new project."""
    new_project = Project(name=project.name, mode=project.mode, metadata=project.metadata)
    db.add(new_project)
    await db.commit()
    await db.refresh(new_project)

    return ProjectResponse(
        id=new_project.id,
        name=new_project.name,
        mode=new_project.mode,
        created_at=new_project.created_at.isoformat(),
        metadata=new_project.metadata,
    )


@router.get("/projects/{project_id}")
async def get_project(project_id: UUID, db: AsyncSession = Depends(get_db)) -> ProjectResponse:
    """Get a project by ID."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return ProjectResponse(
        id=project.id,
        name=project.name,
        mode=project.mode,
        created_at=project.created_at.isoformat(),
        metadata=project.metadata,
    )


@router.put("/projects/{project_id}")
async def update_project(
    project_id: UUID, update: ProjectUpdate, db: AsyncSession = Depends(get_db)
) -> ProjectResponse:
    """Update a project."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if update.name is not None:
        project.name = update.name
    if update.mode is not None:
        project.mode = update.mode
    if update.metadata is not None:
        project.metadata = update.metadata

    await db.commit()
    await db.refresh(project)

    return ProjectResponse(
        id=project.id,
        name=project.name,
        mode=project.mode,
        created_at=project.created_at.isoformat(),
        metadata=project.metadata,
    )


@router.delete("/projects/{project_id}")
async def delete_project(project_id: UUID, db: AsyncSession = Depends(get_db)):
    """Delete a project."""
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()

    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    await db.delete(project)
    await db.commit()

    return {"message": "Project deleted"}
