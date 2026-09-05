"""Deterministic security and fraud-risk evidence aggregation."""


def build_security_evidence(text_guard, image_guard, video_guard, temporal_analysis, image_authenticity=None, video_authenticity=None):
    results = [("text_guard", text_guard), ("image_guard", image_guard), ("video_guard", video_guard), ("temporal_guard", temporal_analysis)]
    triggered = [name for name, result in results if result and (result.get("prediction") == 1 or result.get("is_prompt_injection") or result.get("is_temporal_attack") or result.get("suspicious_frames", 0) > 0)]
    image_synthetic = bool(image_authenticity and image_authenticity.get("synthetic_media_detected")); video_synthetic = bool(video_authenticity and video_authenticity.get("synthetic_media_detected"))
    if image_synthetic: triggered.append("image_authenticity")
    if video_synthetic: triggered.append("video_authenticity")
    return {"ai_security": {"prompt_injection_detected": any(name in triggered for name in ("text_guard", "image_guard", "temporal_guard")), "triggered_guards": [name for name in triggered if name in ("text_guard", "image_guard", "temporal_guard")]}, "fraud_risk": {"synthetic_image_detected": image_synthetic, "synthetic_video_detected": video_synthetic, "image_authenticity": image_authenticity, "video_authenticity": video_authenticity, "evidence_authenticity_flag": image_synthetic or video_synthetic}, "security_flag": bool(triggered), "triggered_guards": triggered, "deterministic_route": "HUMAN_REVIEW" if triggered else "CONTINUE"}


def aggregate(text_result, image_result, temporal_result, llm_result=None):
    evidence = build_security_evidence(text_result, image_result, None, temporal_result)
    return {"security_flag": evidence["security_flag"], "reason": "Prompt-injection signal detected in one or more modalities." if evidence["security_flag"] else "No prompt-injection signal detected in the submitted modalities.", "route": evidence["deterministic_route"]}
