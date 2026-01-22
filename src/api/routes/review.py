"""Human review routes."""
from fastapi import APIRouter, HTTPException
from typing import List
from src.models.schemas import ReviewRequest, HumanDecisionRequest
from src.governance.logger import GovernanceLogger

router = APIRouter()
governance_logger = GovernanceLogger()


@router.get("/reviews", response_model=List[ReviewRequest])
async def list_reviews() -> List[ReviewRequest]:
    """
    List all pending review requests.

    Returns:
        List of pending review requests
    """
    try:
        return governance_logger.list_pending_reviews()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing reviews: {str(e)}")


@router.get("/reviews/{review_id}", response_model=ReviewRequest)
async def get_review(review_id: int) -> ReviewRequest:
    """
    Get review request by ID.

    Args:
        review_id: Review request ID

    Returns:
        Review request details
    """
    try:
        review_request = governance_logger.get_review_request(review_id)
        if not review_request:
            raise HTTPException(status_code=404, detail="Review request not found")
        return review_request
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting review: {str(e)}")


@router.post("/reviews/{review_id}/decide")
async def submit_human_decision(
    review_id: int,
    request: HumanDecisionRequest
) -> dict:
    """
    Submit human decision for a review.

    Args:
        review_id: Review request ID
        request: Human decision request

    Returns:
        Success confirmation
    """
    try:
        success = governance_logger.submit_human_decision(
            review_id,
            request.decision,
            request.rationale,
            request.reviewer_feedback
        )
        if not success:
            raise HTTPException(status_code=404, detail="Review request not found")
        return {"status": "success", "review_id": review_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting decision: {str(e)}")
