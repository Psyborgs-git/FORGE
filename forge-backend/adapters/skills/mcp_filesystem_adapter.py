"""MCP Filesystem adapter for skills registry."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from core.entities import Skill, SkillManifest, SkillResult
from core.ports import SkillRegistryPort
from db.models import SkillModel


class MCPFilesystemAdapter(SkillRegistryPort):
    """Adapter for filesystem-based skills registry (MCP pattern)."""

    def __init__(self, db_session: AsyncSession) -> None:
        """Initialize MCP filesystem adapter."""
        self.db = db_session
        self.skills_base_dir = Path(os.path.expanduser(settings.FORGE_SKILLS_DIR))

        # Create skills directories if they don't exist
        self.public_dir = self.skills_base_dir / "public"
        self.user_dir = self.skills_base_dir / "user"
        self.private_dir = self.skills_base_dir / "private"

        for directory in [self.public_dir, self.user_dir, self.private_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    async def scan_and_register_skills(self) -> None:
        """Scan all skill directories and register valid skills."""
        for skill_dir in [self.public_dir, self.user_dir, self.private_dir]:
            await self._scan_directory(skill_dir)

    async def _scan_directory(self, directory: Path) -> None:
        """Scan a directory for skills and register them."""
        if not directory.exists():
            return

        for skill_path in directory.iterdir():
            if skill_path.is_dir():
                skill_md = skill_path / "SKILL.md"
                if skill_md.exists():
                    try:
                        manifest = self._parse_skill_manifest(skill_md)
                        await self._upsert_skill(manifest, str(skill_path))
                    except Exception as e:
                        print(f"Error parsing skill {skill_path.name}: {e}")

    def _parse_skill_manifest(self, skill_md: Path) -> SkillManifest:
        """Parse SKILL.md frontmatter to extract manifest."""
        content = skill_md.read_text(encoding="utf-8")

        # Extract YAML frontmatter between --- delimiters
        if not content.startswith("---"):
            raise ValueError("SKILL.md must start with YAML frontmatter")

        parts = content.split("---", 2)
        if len(parts) < 3:
            raise ValueError("Invalid SKILL.md format")

        frontmatter_yaml = parts[1].strip()
        frontmatter = yaml.safe_load(frontmatter_yaml)

        # Check for forge-adapter.json
        skill_dir = skill_md.parent
        adapter_json_path = skill_dir / "forge-adapter.json"
        adapter_config = {}

        if adapter_json_path.exists():
            with open(adapter_json_path, "r") as f:
                adapter_config = json.load(f)

        # Merge adapter config with frontmatter
        manifest_dict = {
            "name": frontmatter.get("name"),
            "description": frontmatter.get("description", ""),
            "version": frontmatter.get("version", "1.0.0"),
            "port": frontmatter.get("port"),
            "author": frontmatter.get("author"),
            "tags": frontmatter.get("tags", []),
            "requires": adapter_config.get("requires", frontmatter.get("requires", {})),
            "config_schema": adapter_config.get("config_schema", {}),
            "capabilities": adapter_config.get("capabilities", frontmatter.get("capabilities", [])),
        }

        if not manifest_dict["name"]:
            raise ValueError("Skill must have a name")

        return SkillManifest(**manifest_dict)

    async def _upsert_skill(self, manifest: SkillManifest, skill_path: str) -> None:
        """Insert or update a skill in the database."""
        # Check if skill already exists
        result = await self.db.execute(
            select(SkillModel).where(SkillModel.name == manifest.name)
        )
        existing_skill = result.scalar_one_or_none()

        manifest_dict = {
            "name": manifest.name,
            "description": manifest.description,
            "version": manifest.version,
            "port": manifest.port,
            "author": manifest.author,
            "tags": manifest.tags,
            "requires": manifest.requires,
            "config_schema": manifest.config_schema,
            "capabilities": manifest.capabilities,
        }

        if existing_skill:
            existing_skill.skill_path = skill_path
            existing_skill.manifest = manifest_dict
            existing_skill.port_binding = manifest.port
        else:
            new_skill = SkillModel(
                name=manifest.name,
                skill_path=skill_path,
                manifest=manifest_dict,
                port_binding=manifest.port,
                is_active=True,
            )
            self.db.add(new_skill)

        await self.db.commit()

    async def list_skills(self) -> list[SkillManifest]:
        """List all registered skills."""
        result = await self.db.execute(select(SkillModel).where(SkillModel.is_active == True))
        skills = result.scalars().all()

        manifests = []
        for skill in skills:
            manifest_dict = skill.manifest
            manifests.append(SkillManifest(**manifest_dict))

        return manifests

    async def get_skill(self, name: str) -> Skill:
        """Get a skill by name."""
        result = await self.db.execute(select(SkillModel).where(SkillModel.name == name))
        skill_model = result.scalar_one_or_none()

        if not skill_model:
            raise SkillNotFoundError(f"Skill '{name}' not found")

        # Read skill content
        skill_path = Path(skill_model.skill_path)
        skill_md = skill_path / "SKILL.md"

        if not skill_md.exists():
            raise SkillNotFoundError(f"SKILL.md not found for '{name}'")

        content = skill_md.read_text(encoding="utf-8")

        manifest = SkillManifest(**skill_model.manifest)

        return Skill(
            manifest=manifest,
            skill_path=str(skill_path),
            content=content,
            is_active=skill_model.is_active,
        )

    async def install_skill(self, path_or_url: str) -> SkillManifest:
        """Install a skill from a local path or URL."""
        source_path = Path(path_or_url)

        # For now, only support local paths
        if not source_path.exists():
            raise SkillInstallError(f"Path does not exist: {path_or_url}")

        if not source_path.is_dir():
            raise SkillInstallError(f"Path is not a directory: {path_or_url}")

        skill_md = source_path / "SKILL.md"
        if not skill_md.exists():
            raise SkillInstallError("No SKILL.md found in directory")

        # Parse manifest to get skill name
        manifest = self._parse_skill_manifest(skill_md)

        # Copy to user directory
        dest_path = self.user_dir / manifest.name

        if dest_path.exists():
            shutil.rmtree(dest_path)

        shutil.copytree(source_path, dest_path)

        # Register in database
        await self._upsert_skill(manifest, str(dest_path))

        return manifest

    async def invoke_skill(self, name: str, context: dict[str, Any]) -> SkillResult:
        """Invoke a skill with given context."""
        try:
            skill = await self.get_skill(name)

            # For Phase 1, we just return the skill content as context injection
            # In later phases, skills can have executable logic

            return SkillResult(
                skill_name=name,
                success=True,
                content=skill.content,
                metadata={
                    "manifest": skill.manifest.__dict__,
                    "context": context,
                },
            )
        except Exception as e:
            return SkillResult(
                skill_name=name,
                success=False,
                error=str(e),
            )

    async def uninstall_skill(self, name: str) -> None:
        """Uninstall a skill."""
        result = await self.db.execute(select(SkillModel).where(SkillModel.name == name))
        skill = result.scalar_one_or_none()

        if not skill:
            raise SkillNotFoundError(f"Skill '{name}' not found")

        # Don't delete from public directory
        skill_path = Path(skill.skill_path)
        if self.public_dir in skill_path.parents:
            raise SkillInstallError("Cannot uninstall built-in skills")

        # Delete from filesystem
        if skill_path.exists():
            shutil.rmtree(skill_path)

        # Delete from database
        await self.db.delete(skill)
        await self.db.commit()


class SkillNotFoundError(Exception):
    """Skill not found error."""

    pass


class SkillInstallError(Exception):
    """Skill installation error."""

    pass
