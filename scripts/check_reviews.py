#!/usr/bin/env python3
"""Check review records in the database."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.database import SessionLocal, ReviewRecord, DecisionRecord
from datetime import datetime

def main():
    session = SessionLocal()
    try:
        # Get all reviews
        all_reviews = session.query(ReviewRecord).all()
        print(f"\nTotal reviews in database: {len(all_reviews)}\n")

        # Group by status
        by_status = {}
        for review in all_reviews:
            status = review.status or "None"
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(review)

        print("Reviews by status:")
        for status, reviews in by_status.items():
            print(f"  {status}: {len(reviews)}")

        print("\n" + "="*80)
        print("Detailed Review Information:")
        print("="*80)

        for review in all_reviews:
            decision = review.decision
            print(f"\nReview ID: {review.id}")
            print(f"  Status: {review.status}")
            print(f"  Decision ID: {review.decision_id}")
            print(f"  Created at: {review.created_at}")
            print(f"  Reviewed at: {review.reviewed_at}")
            print(f"  Human decision action: {review.human_decision_action}")
            print(f"  Human rationale: {review.human_rationale}")
            if decision:
                print(f"  Decision action: {decision.decision_action}")
                print(f"  Requires human review: {decision.requires_human_review}")
                transcript_snippet = decision.transcript[:100] if decision.transcript else "N/A"
                print(f"  Transcript snippet: {transcript_snippet}...")
            print("-" * 80)

        # Check decisions that require human review but don't have reviews
        decisions_needing_review = session.query(DecisionRecord).filter(
            DecisionRecord.requires_human_review == True
        ).all()

        print(f"\n\nDecisions requiring human review: {len(decisions_needing_review)}")
        for decision in decisions_needing_review:
            has_review = decision.review is not None
            review_status = decision.review.status if has_review else "NO REVIEW RECORD"
            print(f"  Decision ID {decision.id}: Review status = {review_status}")

    finally:
        session.close()

if __name__ == "__main__":
    main()
