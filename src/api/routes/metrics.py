"""Metrics routes."""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
from src.governance.metrics import MetricsCalculator

router = APIRouter()
metrics_calculator = MetricsCalculator()


@router.get("/metrics")
async def get_metrics(days: int = Query(7, ge=1, le=30)) -> Dict[str, Any]:
    """
    Get core trust metrics.

    Args:
        days: Number of days to look back (default: 7, max: 30)

    Returns:
        Dictionary of trust metrics
    """
    try:
        metrics = metrics_calculator.calculate_metrics(days=days)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating metrics: {str(e)}")
