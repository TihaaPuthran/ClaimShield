"""Optional Supabase persistence for ClaimShield claim history."""

import os
from datetime import datetime, timedelta, timezone

try:
    from supabase import create_client
except ImportError:  # The API remains usable without the optional database package.
    create_client = None


def _client():
    if create_client is None:
        return None
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
    return create_client(url, key) if url and key else None


def save_claim(**claim):
    try:
        client = _client()
        if not client:
            return {"status": "unavailable", "data": None}
        result = client.table("claims").insert(claim).execute()
        return {"status": "saved", "data": result.data}
    except Exception as error:  # Persistence must never break claim analysis.
        return {"status": "unavailable", "data": None, "error": str(error)}


def get_user_claim_history(user_id):
    try:
        client = _client()
        if not client:
            return []
        result = client.table("claims").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return result.data or []
    except Exception:
        return []


def count_user_claims(user_id):
    return len(get_user_claim_history(user_id))


def get_recent_user_claims(user_id, window_days=None):
    days = window_days or int(os.getenv("REPEAT_CLAIM_WINDOW_DAYS", "30"))
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return [claim for claim in get_user_claim_history(user_id) if _created_at(claim) and _created_at(claim) >= cutoff]


def check_duplicate_claims(user_id, image_hash=None, video_hash=None):
    history = get_user_claim_history(user_id)
    return {
        "image": bool(image_hash and any(item.get("image_hash") == image_hash for item in history)),
        "video": bool(video_hash and any(item.get("video_hash") == video_hash for item in history)),
    }


def _created_at(claim):
    value = claim.get("created_at")
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
