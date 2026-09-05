from pathlib import Path
import logging
import time
import hashlib
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from guards.text_guard import analyze_text
from guards.image_guard import analyze_image
from guards.video_guard import analyze_video
from guards.temporal_aggregator import analyze_temporal
from guards.cross_modal_guard import build_security_evidence
from guards.media_authenticity_guard import analyze_image_authenticity, analyze_video_authenticity
from services.groq_analysis import generate_security_explanation
from services.database_service import get_user_claim_history, get_recent_user_claims, save_claim
from guards.repeat_claim_guard import analyze_repeat_claim

app = FastAPI()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("claimshield")
allowed_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOADS_DIR = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)


class TextAnalysisRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "ClaimShield Backend Running"}


@app.get("/health")
def health():
    return {"status": "healthy", "system": "ClaimShield"}


@app.post("/guards/text/analyze")
def analyze_claim_text(request: TextAnalysisRequest):
    try:
        return analyze_text(request.text)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/claims/submit")
async def submit_claim(
    user_id: str = Form(...),
    claim_id: str = Form(...),
    claim_description: str = Form(...),
    claim_amount: float = Form(...),
    image: UploadFile | None = File(None),
    video: UploadFile | None = File(None),
):
    request_started = time.perf_counter()
    if not claim_id.strip():
        raise HTTPException(status_code=422, detail="Claim ID is required")
    if not claim_description.strip():
        raise HTTPException(status_code=422, detail="Claim description is required")
    if claim_amount <= 0:
        raise HTTPException(status_code=422, detail="Claim amount must be greater than 0")
    if image is None and video is None:
        raise HTTPException(status_code=400, detail="At least one evidence file is required: image or video.")
    if not user_id.strip():
        raise HTTPException(status_code=422, detail="User ID is required")

    previous_claims = get_user_claim_history(user_id.strip())
    recent_claims = get_recent_user_claims(user_id.strip())
    image_bytes = await image.read() if image else None
    video_bytes = await video.read() if video else None
    image_hash = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None
    video_hash = hashlib.sha256(video_bytes).hexdigest() if video_bytes else None
    repeat_claim_analysis = analyze_repeat_claim(user_id, claim_description, recent_claims, image_hash, video_hash)

    try:
        stage_started = time.perf_counter()
        logger.info("[ClaimShield] Text Guard start")
        text_guard = analyze_text(claim_description)
        logger.info("[ClaimShield] Text Guard end: %.2fs", time.perf_counter() - stage_started)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    image_filename = Path(image.filename).name if image else None
    video_filename = Path(video.filename).name if video else None
    image_guard = None
    video_guard = None
    temporal = None
    image_authenticity = None
    video_authenticity = None
    if image:
        stage_started = time.perf_counter()
        logger.info("[ClaimShield] Image Guard start")
        image_path = UPLOADS_DIR / image_filename
        image_path.write_bytes(image_bytes)
        image_guard = analyze_image(image_path)
        image_authenticity = analyze_image_authenticity(image_path)
        logger.info("[ClaimShield] Image Guard end: %.2fs", time.perf_counter() - stage_started)
    if video:
        stage_started = time.perf_counter()
        video_path = UPLOADS_DIR / video_filename
        video_path.write_bytes(video_bytes)
        video_guard = analyze_video(video_path)
        logger.info("[ClaimShield] Video processing end: %.2fs", time.perf_counter() - stage_started)
        temporal_started = time.perf_counter()
        temporal = analyze_temporal(video_guard["frame_results"])
        video_authenticity = analyze_video_authenticity(video_path)
        logger.info("[ClaimShield] Temporal analysis: %.2fs", time.perf_counter() - temporal_started)
    llm_started = time.perf_counter()
    security_evidence = build_security_evidence(text_guard, image_guard, video_guard, temporal, image_authenticity, video_authenticity)
    security_evidence["repeat_claim_analysis"] = repeat_claim_analysis
    llm_analysis = generate_security_explanation(security_evidence)
    logger.info("[ClaimShield] LLM analysis end: %.2fs", time.perf_counter() - llm_started)
    cross_started = time.perf_counter()
    cross_modal = {"security_flag": security_evidence["security_flag"], "reason": "Prompt-injection signal detected in one or more modalities." if security_evidence["security_flag"] else "No prompt-injection signal detected in the submitted modalities.", "route": security_evidence["deterministic_route"]}
    logger.info("[ClaimShield] Cross-modal decision: %.2fs", time.perf_counter() - cross_started)
    database = save_claim(claim_id=claim_id, user_id=user_id, claim_description=claim_description, claim_amount=claim_amount, image_reference=image_filename, video_reference=video_filename, image_hash=image_hash, video_hash=video_hash, text_guard=text_guard, image_guard=image_guard, video_guard=video_guard, temporal_analysis=temporal, media_authenticity={"image": image_authenticity, "video": video_authenticity}, security_evidence=security_evidence, llm_analysis=llm_analysis, final_route=security_evidence["deterministic_route"], security_flag=security_evidence["security_flag"])
    logger.info("[ClaimShield] Database save: %s", database["status"])
    logger.info("[ClaimShield] Total request duration: %.2fs", time.perf_counter() - request_started)

    return {
        "message": "Claim received successfully",
        "claim_id": claim_id,
        "user_id": user_id,
        "claim_description": claim_description,
        "claim_amount": claim_amount,
        "image_filename": image_filename,
        "video_filename": video_filename,
        "status": "received",
        "text_guard": text_guard,
        "image_guard": image_guard,
        "video_guard": video_guard,
        "temporal_analysis": temporal,
        "llm_analysis": llm_analysis,
        "image_authenticity": image_authenticity,
        "video_authenticity": video_authenticity,
        "security_evidence": security_evidence,
        "repeat_claim_analysis": repeat_claim_analysis,
        "database_status": database["status"],
        "cross_modal_decision": cross_modal,
        "final_decision": {"security_flag": security_evidence["security_flag"], "route": security_evidence["deterministic_route"]},
    }


@app.get("/users/{user_id}/claims")
def user_claim_history(user_id: str):
    return {"user_id": user_id, "claims": get_user_claim_history(user_id)}


@app.get("/users/{user_id}/claims/summary")
def user_claim_summary(user_id: str):
    history = get_user_claim_history(user_id)
    recent = get_recent_user_claims(user_id)
    return {"user_id": user_id, "total_claims": len(history), "claims_last_30_days": len(recent), "human_review_count": sum(item.get("final_route") == "HUMAN_REVIEW" for item in history), "latest_claim": history[0] if history else None}
