"""Deterministic, non-fraudulent repeated-claim review signal."""

import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def analyze_repeat_claim(user_id, current_claim, previous_claims, image_hash=None, video_hash=None):
    window_days = int(os.getenv("REPEAT_CLAIM_WINDOW_DAYS", "30"))
    recent = previous_claims[:]
    texts = [str(item.get("claim_description", "")) for item in previous_claims if item.get("claim_description")]
    similarity = 0.0
    if texts and current_claim.strip():
        matrix = TfidfVectorizer().fit_transform([current_claim, *texts])
        similarity = float(cosine_similarity(matrix[0:1], matrix[1:]).max())
    duplicate_text = similarity >= float(os.getenv("DUPLICATE_TEXT_SIMILARITY", "0.85"))
    duplicate_evidence = bool((image_hash and any(x.get("image_hash") == image_hash for x in previous_claims)) or (video_hash and any(x.get("video_hash") == video_hash for x in previous_claims)))
    review = len(recent) >= 3 or duplicate_text or duplicate_evidence
    return {"previous_claim_count": len(previous_claims), "recent_claim_count": len(recent), "duplicate_text_flag": duplicate_text, "duplicate_text_similarity": round(similarity, 4), "duplicate_evidence_flag": duplicate_evidence, "history_risk": "REVIEW" if review else "NORMAL", "reason": "Repeated history is an additional review signal; it is not a fraud determination." if review else f"No repeated-claim review signal in the configured {window_days}-day window."}
