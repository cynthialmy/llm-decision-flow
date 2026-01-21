"""Content analysis routes."""
import logging
from fastapi import APIRouter, HTTPException
from src.models.schemas import AnalysisRequest, AnalysisResponse
from src.orchestrator.decision_orchestrator import DecisionOrchestrator
from src.governance.logger import GovernanceLogger

logger = logging.getLogger(__name__)

router = APIRouter()
orchestrator = DecisionOrchestrator()
governance_logger = GovernanceLogger()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_content(request: AnalysisRequest) -> AnalysisResponse:
    """
    Analyze content transcript and return decision.

    Args:
        request: Analysis request with transcript

    Returns:
        Analysis response with decision and all intermediate results
    """
    try:
        # Execute agent pipeline
        analysis_response = orchestrator.analyze(request.transcript)

        # Log decision for governance
        decision_id = governance_logger.log_decision(analysis_response, request.transcript)

        # Set review request ID if escalated
        if analysis_response.decision.requires_human_review:
            # Find the review request ID
            pending_reviews = governance_logger.list_pending_reviews()
            if pending_reviews:
                # Get the most recent one (should be ours)
                analysis_response.review_request_id = pending_reviews[-1].id

        return analysis_response
    except Exception as e:
        logger.error(f"Error analyzing content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing content: {str(e)}")
