"""Hardware API router."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import get_hardware_advisor
from services.hardware_advisor import HardwareAdvisor

router = APIRouter()


@router.get("/profile")
async def get_hardware_profile(
    advisor: HardwareAdvisor = Depends(get_hardware_advisor),
):
    """Get current hardware profile."""
    profile = advisor.get_hardware_profile()
    return profile.to_dict()


@router.get("/feasible-ops")
async def get_feasible_operations(
    advisor: HardwareAdvisor = Depends(get_hardware_advisor),
):
    """Get feasible operations based on hardware."""
    profile = advisor.get_hardware_profile()
    feasible_ops = advisor.get_feasible_operations(profile)
    return feasible_ops.to_dict()
