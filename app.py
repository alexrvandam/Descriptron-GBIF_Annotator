"""
Descriptron GBIF Web Annotator — SAM2.1 Encoder Backend
========================================================
ONNX Split Architecture:
  - This server: encodes images → returns embedding tensors
  - Browser: runs ONNX decoder locally → instant mask prediction

Endpoints:
  POST /encode              — encode image, return embedding as binary
  GET  /proxy-image         — CORS proxy for GBIF images
  GET  /onnx/decoder.onnx   — serve the SAM2.1-tiny decoder ONNX model
  GET  /health              — health check
"""

import io
import os
import time
import hashlib
import logging
import struct
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel
import httpx
import pathlib

# ─── Configuration ────────────────────────────────────────────────────
SAM2_CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", "/app/checkpoints/sam2.1_hiera_tiny.pt")
SAM2_CONFIG = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml")
ONNX_DECODER_PATH = os.environ.get("ONNX_DECODER_PATH", "/app/onnx/sam2.1_hiera_tiny_decoder.onnx")
MAX_CACHE_SIZE = int(os.environ.get("MAX_CACHE_SIZE", "100"))
MAX_IMAGE_SIZE = int(os.environ.get("MAX_IMAGE_SIZE", "1024"))  # SAM2 input size
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gbif-sam2")

# ─── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="Descriptron GBIF SAM2 Encoder API", version="2.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)

# ─── SAM2 Model ──────────────────────────────────────────────────────
sam2_model = None
image_encoder = None
device = "cpu"

def load_sam2():
    """Load SAM2.1-tiny image encoder only."""
    global sam2_model, image_encoder, device
    
    from sam2.build_sam import build_sam2
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading SAM2.1-tiny on {device}...")
    
    if not os.path.exists(SAM2_CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {SAM2_CHECKPOINT}")
    
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    sam2_model.eval()
    
    logger.info(f"SAM2.1-tiny loaded on {device}")


# ─── Embedding Cache ─────────────────────────────────────────────────
class EmbeddingCache:
    """LRU cache for image encoder outputs."""
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.url_map = {}
    
    def _url_hash(self, url):
        return hashlib.sha256(url.encode()).hexdigest()[:16]
    
    def get_by_url(self, url):
        h = self._url_hash(url)
        eid = self.url_map.get(h)
        if eid and eid in self.cache:
            self.cache.move_to_end(eid)
            return eid, self.cache[eid]
        return None, None
    
    def put(self, eid, url, data):
        if len(self.cache) >= self.max_size:
            old_key, _ = self.cache.popitem(last=False)
            self.url_map = {k: v for k, v in self.url_map.items() if v != old_key}
        self.cache[eid] = data
        self.url_map[self._url_hash(url)] = eid
    
    def delete(self, eid):
        if eid in self.cache:
            del self.cache[eid]
            self.url_map = {k: v for k, v in self.url_map.items() if v != eid}
            return True
        return False
    
    def stats(self):
        return {"cached": len(self.cache), "max_size": self.max_size}

cache = EmbeddingCache(max_size=MAX_CACHE_SIZE)


# ─── Image preprocessing (match SAM2's transform) ────────────────────
def preprocess_image(img_pil, target_length=1024):
    """
    Resize + normalize image exactly as SAM2 does internally.
    Returns: (input_tensor [1,3,1024,1024], original_size, input_size)
    """
    w, h = img_pil.size
    scale = target_length / max(w, h)
    new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
    
    img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
    img_np = np.array(img_resized).astype(np.float32)  # [H, W, 3]
    
    # Normalize with ImageNet stats (same as SAM2)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img_np = (img_np - mean) / std
    
    # Pad to 1024x1024
    padded = np.zeros((target_length, target_length, 3), dtype=np.float32)
    padded[:new_h, :new_w, :] = img_np
    
    # [H,W,3] -> [1,3,H,W]
    tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0)
    
    return tensor, (h, w), (new_h, new_w)


def run_encoder(img_tensor):
    """
    Run SAM2.1 image encoder, return the intermediate features
    needed by the decoder.
    
    Returns dict with:
      - image_embed: [1, 256, 64, 64] high-res feature map
      - high_res_feats_0: [1, 32, 256, 256] 
      - high_res_feats_1: [1, 64, 128, 128]
    """
    with torch.inference_mode():
        img_tensor = img_tensor.to(device)
        
        # Use SAM2's image encoder
        backbone_out = sam2_model.forward_image(img_tensor)
        
        # Process through the model's feature preparation
        _, vision_feats, _, _ = sam2_model._prepare_backbone_features(backbone_out)
        
        # Get the features in the format the decoder expects
        # vision_feats is a list of tensors at different scales
        # For SAM2, we need to reshape these for the mask decoder
        
        # The image encoder output that the ONNX decoder needs:
        # We need to run through sam2_model's _prepare_memory_conditioned_features
        # or equivalent to get the right format.
        
        # Simpler approach: use the SAM2ImagePredictor to get features
        # then extract what the ONNX decoder needs
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor = SAM2ImagePredictor(sam2_model)
        
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # Undo normalization for set_image
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        img_np = (img_np * std + mean).clip(0, 255).astype(np.uint8)
        
        predictor.set_image(img_np)
        
        # Extract the features the ONNX decoder needs
        features = predictor._features
        
        # The key tensors for the decoder:
        result = {}
        
        # Image embeddings (the main feature map)
        result["image_embed"] = features["image_embed"].cpu().numpy()
        
        # High-resolution features at two scales
        high_res = features.get("high_res_feats", [])
        if len(high_res) >= 2:
            result["high_res_feats_0"] = high_res[0].cpu().numpy()
            result["high_res_feats_1"] = high_res[1].cpu().numpy()
        
        return result


def pack_embeddings(embed_dict, orig_size, input_size):
    """
    Pack embedding tensors into a binary format for efficient transfer.
    
    Format:
      [header: 32 bytes]
        - magic: 4 bytes "SAM2"
        - version: 4 bytes (uint32 = 2)
        - orig_h, orig_w: 4+4 bytes (uint32)
        - input_h, input_w: 4+4 bytes (uint32)
        - num_tensors: 4 bytes (uint32)
        - reserved: 4 bytes
      [tensor entries]
        For each tensor:
          - name_len: 4 bytes (uint32)
          - name: name_len bytes (utf-8)
          - ndim: 4 bytes (uint32)
          - shape: ndim * 4 bytes (uint32 each)
          - dtype: 4 bytes (0=float32, 1=float16)
          - data_len: 4 bytes (uint32)
          - data: data_len bytes
    """
    buf = io.BytesIO()
    
    # Header
    buf.write(b"SAM2")                                    # magic
    buf.write(struct.pack("<I", 2))                       # version
    buf.write(struct.pack("<II", orig_size[0], orig_size[1]))   # orig H, W
    buf.write(struct.pack("<II", input_size[0], input_size[1])) # input H, W
    buf.write(struct.pack("<I", len(embed_dict)))         # num tensors
    buf.write(struct.pack("<I", 0))                       # reserved
    
    for name, arr in embed_dict.items():
        # Convert to float16 for smaller transfer (decoder handles this)
        arr_f16 = arr.astype(np.float16)
        data = arr_f16.tobytes()
        
        name_bytes = name.encode("utf-8")
        buf.write(struct.pack("<I", len(name_bytes)))
        buf.write(name_bytes)
        buf.write(struct.pack("<I", arr_f16.ndim))
        for s in arr_f16.shape:
            buf.write(struct.pack("<I", s))
        buf.write(struct.pack("<I", 1))  # dtype: 1 = float16
        buf.write(struct.pack("<I", len(data)))
        buf.write(data)
    
    return buf.getvalue()


# ─── Startup ──────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    try:
        load_sam2()
    except Exception as e:
        logger.error(f"Failed to load SAM2: {e}")
        logger.warning("Starting WITHOUT encoder — only proxy will work")


# ─── Image fetcher ────────────────────────────────────────────────────
async def fetch_image(url: str) -> Image.Image:
    # Local uploads: read directly from /tmp/uploads/ (no HTTP needed)
    if url.startswith("/uploads/"):
        local_path = pathlib.Path("/tmp") / url.lstrip("/")
        if not local_path.exists():
            raise HTTPException(404, f"Uploaded file not found: {url}")
        try:
            return Image.open(str(local_path)).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"Invalid uploaded image: {e}")
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Failed to fetch image: {e}")
    try:
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


# ─── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    onnx_exists = os.path.exists(ONNX_DECODER_PATH)
    return {
        "status": "ok",
        "encoder_loaded": sam2_model is not None,
        "device": device,
        "onnx_decoder_available": onnx_exists,
        "cache": cache.stats(),
    }


@app.get("/proxy-image")
async def proxy_image(url: str = Query(...)):
    """CORS proxy for GBIF images — solves cross-origin canvas issues."""
    if not url.startswith("http"):
        raise HTTPException(400, "Invalid URL")
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(502, f"Fetch failed: {e}")
    return StreamingResponse(
        io.BytesIO(resp.content),
        media_type=resp.headers.get("content-type", "image/jpeg"),
        headers={"Cache-Control": "public, max-age=86400", "Access-Control-Allow-Origin": "*"},
    )


class EncodeRequest(BaseModel):
    image_url: str


@app.post("/encode")
async def encode_image(req: EncodeRequest):
    """
    Encode an image with SAM2.1 encoder.
    
    Returns the embedding tensors as a binary blob that the browser's
    ONNX decoder can consume directly. This is the heavy computation
    (~15-30s CPU). After this, all prompting happens client-side.
    
    Response: binary data (application/octet-stream) with packed tensors.
    Use /encode-json for a JSON metadata response without the binary.
    """
    if sam2_model is None:
        raise HTTPException(503, "Encoder not loaded")
    
    # Check cache
    existing_id, existing_data = cache.get_by_url(req.image_url)
    if existing_id and existing_data:
        logger.info(f"Cache hit: {req.image_url[:60]}... -> {existing_id}")
        return StreamingResponse(
            io.BytesIO(existing_data["packed"]),
            media_type="application/octet-stream",
            headers={
                "X-Embedding-Id": existing_id,
                "X-Cached": "true",
                "X-Orig-Width": str(existing_data["orig_size"][1]),
                "X-Orig-Height": str(existing_data["orig_size"][0]),
                "Access-Control-Expose-Headers": "X-Embedding-Id, X-Cached, X-Orig-Width, X-Orig-Height",
                "Access-Control-Allow-Origin": "*",
            },
        )
    
    # Fetch image
    t0 = time.time()
    img = await fetch_image(req.image_url)
    orig_w, orig_h = img.size
    t_fetch = time.time() - t0
    
    # Resize to max dimension
    if max(orig_w, orig_h) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(orig_w, orig_h)
        img = img.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
    
    # Encode
    t1 = time.time()
    img_tensor, orig_size, input_size = preprocess_image(img)
    embed_dict = run_encoder(img_tensor)
    t_encode = time.time() - t1
    
    # Pack into binary
    packed = pack_embeddings(embed_dict, (orig_h, orig_w), input_size)
    
    # Cache
    import uuid
    eid = str(uuid.uuid4())[:12]
    cache.put(eid, req.image_url, {
        "packed": packed,
        "orig_size": (orig_h, orig_w),
        "input_size": input_size,
    })
    
    logger.info(f"Encoded {req.image_url[:60]}... -> {eid} "
                f"({orig_w}x{orig_h}, fetch={t_fetch:.1f}s, encode={t_encode:.1f}s, "
                f"packed={len(packed)/1024:.0f}KB)")
    
    return StreamingResponse(
        io.BytesIO(packed),
        media_type="application/octet-stream",
        headers={
            "X-Embedding-Id": eid,
            "X-Cached": "false",
            "X-Orig-Width": str(orig_w),
            "X-Orig-Height": str(orig_h),
            "X-Fetch-Time": f"{t_fetch:.2f}",
            "X-Encode-Time": f"{t_encode:.2f}",
            "Access-Control-Expose-Headers": "X-Embedding-Id, X-Cached, X-Orig-Width, X-Orig-Height, X-Fetch-Time, X-Encode-Time",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/encode-json")
async def encode_image_json(req: EncodeRequest):
    """
    Same as /encode but returns JSON metadata only (no binary).
    The binary embedding can be fetched separately via /embedding/{id}.
    Useful for checking cache status without downloading the full blob.
    """
    if sam2_model is None:
        raise HTTPException(503, "Encoder not loaded")
    
    existing_id, existing_data = cache.get_by_url(req.image_url)
    if existing_id:
        return {
            "embedding_id": existing_id,
            "cached": True,
            "orig_size": {"w": existing_data["orig_size"][1], "h": existing_data["orig_size"][0]},
            "packed_size_kb": round(len(existing_data["packed"]) / 1024, 1),
        }
    
    # Must encode first — redirect to /encode
    raise HTTPException(404, "Not cached. POST to /encode to generate embedding.")


@app.get("/embedding/{embedding_id}")
async def get_embedding(embedding_id: str):
    """Download a previously cached embedding by ID."""
    data = None
    for eid, d in cache.cache.items():
        if eid == embedding_id:
            data = d
            break
    if not data:
        raise HTTPException(404, "Embedding not found")
    
    return StreamingResponse(
        io.BytesIO(data["packed"]),
        media_type="application/octet-stream",
        headers={
            "X-Embedding-Id": embedding_id,
            "X-Orig-Width": str(data["orig_size"][1]),
            "X-Orig-Height": str(data["orig_size"][0]),
            "Access-Control-Expose-Headers": "X-Embedding-Id, X-Orig-Width, X-Orig-Height",
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.get("/onnx/decoder.onnx")
async def serve_decoder_onnx():
    """
    Serve the SAM2.1-tiny decoder ONNX model for client-side inference.
    The browser downloads this once and caches it.
    ~10MB file, runs at ~10ms per prompt in the browser.
    """
    if not os.path.exists(ONNX_DECODER_PATH):
        raise HTTPException(404, 
            "Decoder ONNX not found. Export it with: "
            "python export_onnx_decoder.py --checkpoint sam2.1_hiera_tiny.pt --output decoder.onnx")
    
    return FileResponse(
        ONNX_DECODER_PATH,
        media_type="application/octet-stream",
        filename="sam2.1_hiera_tiny_decoder.onnx",
        headers={
            "Cache-Control": "public, max-age=604800",  # Cache 7 days
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.delete("/cache/{embedding_id}")
async def clear_cache(embedding_id: str):
    if cache.delete(embedding_id):
        return {"deleted": embedding_id}
    raise HTTPException(404, "Not found")


@app.get("/cache/stats")
async def cache_stats():
    return cache.stats()


# ─── Server-side decode fallback ─────────────────────────────────────
from typing import List, Optional
import base64
import cv2

class PointPrompt(BaseModel):
    x: float
    y: float
    label: int

class BboxPrompt(BaseModel):
    x: float
    y: float
    w: float
    h: float

class DecodeRequest(BaseModel):
    embedding_id: Optional[str] = None
    image_url: Optional[str] = None
    points: List[PointPrompt] = []
    bboxes: List[BboxPrompt] = []

@app.post("/decode")
async def decode_mask(req: DecodeRequest):
    if sam2_model is None:
        raise HTTPException(503, "Model not loaded")
    if not req.image_url:
        raise HTTPException(400, "image_url required")
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    img = await fetch_image(req.image_url)
    img_np = np.array(img.convert("RGB"))
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(img_np)
    kwargs = {}
    if len(req.bboxes) == 1:
        b = req.bboxes[0]
        kwargs["box"] = np.array([b.x, b.y, b.x + b.w, b.y + b.h], dtype=np.float32)
    if req.points:
        kwargs["point_coords"] = np.array([[p.x, p.y] for p in req.points], dtype=np.float32)
        kwargs["point_labels"] = np.array([p.label for p in req.points], dtype=np.int32)
    if not kwargs:
        raise HTTPException(400, "No prompts provided")
    masks, scores, _ = predictor.predict(multimask_output=True, **kwargs)
    best_idx = int(np.argmax(scores))
    mask = masks[best_idx].astype(np.uint8)
    score = float(scores[best_idx])
    mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask_b64 = base64.b64encode(mask_256.tobytes()).decode("ascii")
    logger.info(f"Decoded mask: score={score:.3f}")
    return {"mask_w": 256, "mask_h": 256, "mask_b64": mask_b64, "score": score}




# ─── Usage tracking API endpoints ────────────────────────────────────
from pydantic import BaseModel as _BaseModel
from typing import Optional as _Optional
import json as _json
import hashlib as _hashlib

class LoginRequest(_BaseModel):
    username: str
    display_name: _Optional[str] = None
    orcid: _Optional[str] = None
    institution: _Optional[str] = None
    email: _Optional[str] = None

class EventRequest(_BaseModel):
    event_type: str  # encode, decode, export_coco, export_dwc, export_csv, etc.
    details: _Optional[str] = None
    session_token: _Optional[str] = None

@app.post("/api/login")
async def api_login(req: LoginRequest, request: Request):
    """Register or update a user and create a session."""
    username = req.username.strip().lower()
    if not username or len(username) < 2:
        raise HTTPException(400, "Username must be at least 2 characters")

    with _get_db() as db:
        # Upsert user
        existing = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            user_id = existing['id']
            db.execute("""UPDATE users SET
                display_name = COALESCE(?, display_name),
                orcid = COALESCE(?, orcid),
                institution = COALESCE(?, institution),
                email = COALESCE(?, email),
                last_seen = datetime('now')
                WHERE id = ?""",
                (req.display_name, req.orcid, req.institution, req.email, user_id))
        else:
            cur = db.execute("""INSERT INTO users (username, display_name, orcid, institution, email)
                VALUES (?, ?, ?, ?, ?)""",
                (username, req.display_name, req.orcid, req.institution, req.email))
            user_id = cur.lastrowid

        # Create session
        token = _hashlib.sha256(f"{username}-{_time.time()}-{id(request)}".encode()).hexdigest()[:32]
        ua = request.headers.get('user-agent', '')[:200]
        ip = request.client.host if request.client else ''
        db.execute("""INSERT INTO sessions (user_id, session_token, user_agent, ip_address)
            VALUES (?, ?, ?, ?)""", (user_id, token, ua, ip))

        # Log login event
        db.execute("""INSERT INTO events (user_id, event_type, details)
            VALUES (?, 'login', ?)""",
            (user_id, _json.dumps({"ip": ip, "institution": req.institution})))

    logger.info(f"Login: {username} ({req.institution or 'no institution'})")
    return {"session_token": token, "username": username, "user_id": user_id}

@app.post("/api/log-event")
async def api_log_event(req: EventRequest, request: Request):
    """Log a usage event."""
    valid_events = {'encode', 'decode', 'export_coco', 'export_dwc', 'export_csv',
                    'load_gbif', 'load_upload', 'segment', 'add_keypoint', 'add_bbox',
                    'template_change', 'import_coco', 'page_view'}
    if req.event_type not in valid_events:
        raise HTTPException(400, f"Unknown event type: {req.event_type}")

    with _get_db() as db:
        user_id = None
        session_id = None

        if req.session_token:
            sess = db.execute("""SELECT s.id, s.user_id FROM sessions s
                WHERE s.session_token = ?""", (req.session_token,)).fetchone()
            if sess:
                session_id = sess['id']
                user_id = sess['user_id']
                db.execute("UPDATE sessions SET last_active = datetime('now') WHERE id = ?",
                    (session_id,))
                db.execute("UPDATE users SET last_seen = datetime('now') WHERE id = ?",
                    (user_id,))

        db.execute("""INSERT INTO events (session_id, user_id, event_type, details)
            VALUES (?, ?, ?, ?)""",
            (session_id, user_id, req.event_type, req.details))

    return {"ok": True}

@app.get("/api/usage-stats")
async def api_usage_stats():
    """Usage statistics summary for grant reporting."""
    with _get_db() as db:
        stats = {}

        # Total users
        r = db.execute("SELECT COUNT(*) as n FROM users").fetchone()
        stats['total_users'] = r['n']

        # Active users (last 30 days)
        r = db.execute("""SELECT COUNT(*) as n FROM users
            WHERE last_seen >= datetime('now', '-30 days')""").fetchone()
        stats['active_users_30d'] = r['n']

        # Total events
        r = db.execute("SELECT COUNT(*) as n FROM events").fetchone()
        stats['total_events'] = r['n']

        # Events by type
        rows = db.execute("""SELECT event_type, COUNT(*) as n
            FROM events GROUP BY event_type ORDER BY n DESC""").fetchall()
        stats['events_by_type'] = {r['event_type']: r['n'] for r in rows}

        # Events last 30 days by type
        rows = db.execute("""SELECT event_type, COUNT(*) as n
            FROM events WHERE timestamp >= datetime('now', '-30 days')
            GROUP BY event_type ORDER BY n DESC""").fetchall()
        stats['events_30d_by_type'] = {r['event_type']: r['n'] for r in rows}

        # Daily event counts (last 90 days)
        rows = db.execute("""SELECT date(timestamp) as day, COUNT(*) as n
            FROM events WHERE timestamp >= datetime('now', '-90 days')
            GROUP BY day ORDER BY day""").fetchall()
        stats['daily_events'] = {r['day']: r['n'] for r in rows}

        # Per-user summary
        rows = db.execute("""SELECT u.username, u.display_name, u.institution,
            u.orcid, u.created_at, u.last_seen,
            COUNT(e.id) as event_count,
            COUNT(DISTINCT date(e.timestamp)) as active_days
            FROM users u LEFT JOIN events e ON e.user_id = u.id
            GROUP BY u.id ORDER BY event_count DESC""").fetchall()
        stats['users'] = [{
            'username': r['username'], 'display_name': r['display_name'],
            'institution': r['institution'], 'orcid': r['orcid'],
            'created_at': r['created_at'], 'last_seen': r['last_seen'],
            'event_count': r['event_count'], 'active_days': r['active_days']
        } for r in rows]

        # Top images encoded
        rows = db.execute("""SELECT details, COUNT(*) as n
            FROM events WHERE event_type = 'encode'
            GROUP BY details ORDER BY n DESC LIMIT 20""").fetchall()
        stats['top_images'] = [{
            'image': r['details'], 'count': r['n']
        } for r in rows]

    return stats

@app.get("/api/usage-csv")
async def api_usage_csv():
    """Download raw event log as CSV."""
    import csv as _csv
    import io as _io
    with _get_db() as db:
        rows = db.execute("""SELECT e.timestamp, u.username, u.institution,
            e.event_type, e.details
            FROM events e LEFT JOIN users u ON e.user_id = u.id
            ORDER BY e.timestamp DESC""").fetchall()

    output = _io.StringIO()
    writer = _csv.writer(output)
    writer.writerow(['timestamp', 'username', 'institution', 'event_type', 'details'])
    for r in rows:
        writer.writerow([r['timestamp'], r['username'], r['institution'],
                        r['event_type'], r['details']])

    from starlette.responses import Response
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=gbif_annotator_usage.csv"}
    )



# ─── OAuth login for GBIF and iNaturalist ────────────────────────────
import urllib.parse as _urllib_parse

OAUTH_BASE_URL = os.environ.get("OAUTH_BASE_URL", "http://46.225.84.116:8100")
INAT_CLIENT_ID = os.environ.get("INAT_CLIENT_ID", "")
INAT_CLIENT_SECRET = os.environ.get("INAT_CLIENT_SECRET", "")

@app.get("/auth/gbif")
async def auth_gbif(redirect: str = "/"):
    """GBIF login — redirect to a server-side login form since GBIF
    doesn't have OAuth2. We verify credentials via their API."""
    # Return a minimal login form that posts to /auth/gbif/verify
    form_html = f"""<!DOCTYPE html><html><head><title>GBIF Login</title>
    <style>body{{font-family:system-ui;max-width:360px;margin:60px auto;padding:20px}}
    h2{{color:#509e2f}}input{{width:100%;padding:8px;margin:6px 0 12px;border:1px solid #ddd;border-radius:4px;box-sizing:border-box}}
    button{{width:100%;padding:10px;background:#509e2f;color:white;border:none;border-radius:4px;font-size:14px;cursor:pointer}}
    button:hover{{background:#3d7a24}}.note{{color:#888;font-size:12px;margin-top:12px}}</style></head>
    <body><h2>🌿 Login with GBIF</h2>
    <form method="POST" action="/auth/gbif/verify?redirect={_urllib_parse.quote(redirect, safe='')}">
    <label>GBIF Username</label><input name="username" required autofocus>
    <label>GBIF Password</label><input name="password" type="password" required>
    <button type="submit">Sign in</button>
    <p class="note">Your credentials are verified against the GBIF API and not stored.</p>
    </form></body></html>"""
    from starlette.responses import HTMLResponse
    return HTMLResponse(form_html)

@app.post("/auth/gbif/verify")
async def auth_gbif_verify(request: Request, redirect: str = "/"):
    """Verify GBIF credentials via their API and create a session."""
    form = await request.form()
    username = form.get("username", "").strip()
    password = form.get("password", "")

    if not username or not password:
        from starlette.responses import HTMLResponse
        return HTMLResponse("<h3>Missing credentials</h3><a href='javascript:history.back()'>Back</a>", 400)

    # Verify against GBIF API
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(
                f"https://api.gbif.org/v1/user/login",
                auth=(username, password),
                timeout=10.0
            )
            if r.status_code == 200:
                user_data = r.json() if r.headers.get('content-type','').startswith('application/json') else {}
                display_name = user_data.get("firstName", "") + " " + user_data.get("lastName", "")
                display_name = display_name.strip() or username
                institution = user_data.get("settings", {}).get("country", "")
                email = user_data.get("email", "")

                # Create session in our DB
                with _get_db() as db:
                    existing = db.execute("SELECT id FROM users WHERE username = ?", (f"gbif:{username}",)).fetchone()
                    if existing:
                        user_id = existing['id']
                        db.execute("UPDATE users SET display_name=?, last_seen=datetime('now') WHERE id=?",
                            (display_name, user_id))
                    else:
                        cur = db.execute("INSERT INTO users (username, display_name, institution, email) VALUES (?,?,?,?)",
                            (f"gbif:{username}", display_name, institution, email))
                        user_id = cur.lastrowid

                    token = _hashlib.sha256(f"gbif-{username}-{_time.time()}".encode()).hexdigest()[:32]
                    ip = request.client.host if request.client else ''
                    db.execute("INSERT INTO sessions (user_id, session_token, user_agent, ip_address) VALUES (?,?,?,?)",
                        (user_id, token, request.headers.get('user-agent','')[:200], ip))
                    db.execute("INSERT INTO events (user_id, event_type, details) VALUES (?, 'login', ?)",
                        (user_id, _json.dumps({"provider": "gbif", "gbif_username": username})))

                # Redirect back to annotator with auth params
                sep = '&' if '?' in redirect else '?'
                return_url = f"{redirect}{sep}auth_token={token}&auth_user={_urllib_parse.quote(display_name)}&auth_provider=GBIF&auth_institution={_urllib_parse.quote(institution)}"
                from starlette.responses import RedirectResponse
                return RedirectResponse(return_url, status_code=302)
            else:
                from starlette.responses import HTMLResponse
                return HTMLResponse("<h3>⚠️ Invalid GBIF credentials</h3><p>Please check your username and password.</p><a href='javascript:history.back()'>Try again</a>", 401)
        except Exception as e:
            logger.error(f"GBIF auth error: {e}")
            from starlette.responses import HTMLResponse
            return HTMLResponse(f"<h3>Error connecting to GBIF</h3><p>{str(e)}</p><a href='javascript:history.back()'>Back</a>", 500)

@app.get("/auth/inat")
async def auth_inat(redirect: str = "/"):
    """Redirect to iNaturalist OAuth2 authorization."""
    if not INAT_CLIENT_ID:
        from starlette.responses import HTMLResponse
        return HTMLResponse("<h3>iNaturalist login not configured</h3><p>INAT_CLIENT_ID env var not set.</p>", 503)

    callback_url = f"{OAUTH_BASE_URL}/auth/inat/callback"
    # Store redirect URL in a cookie for the callback
    auth_url = (f"https://www.inaturalist.org/oauth/authorize"
                f"?client_id={INAT_CLIENT_ID}"
                f"&redirect_uri={_urllib_parse.quote(callback_url)}"
                f"&response_type=code")

    from starlette.responses import RedirectResponse
    resp = RedirectResponse(auth_url, status_code=302)
    resp.set_cookie("oauth_redirect", redirect, max_age=600, httponly=True)
    return resp

@app.get("/auth/inat/callback")
async def auth_inat_callback(code: str, request: Request):
    """Handle iNaturalist OAuth2 callback."""
    redirect = request.cookies.get("oauth_redirect", "/")
    callback_url = f"{OAUTH_BASE_URL}/auth/inat/callback"

    async with httpx.AsyncClient() as client:
        # Exchange code for token
        token_resp = await client.post("https://www.inaturalist.org/oauth/token", data={
            "client_id": INAT_CLIENT_ID,
            "client_secret": INAT_CLIENT_SECRET,
            "code": code,
            "redirect_uri": callback_url,
            "grant_type": "authorization_code"
        }, timeout=10.0)

        if token_resp.status_code != 200:
            from starlette.responses import HTMLResponse
            return HTMLResponse(f"<h3>iNaturalist auth failed</h3><p>{token_resp.text}</p>", 401)

        access_token = token_resp.json().get("access_token")

        # Fetch user profile
        me_resp = await client.get("https://api.inaturalist.org/v1/users/me",
            headers={"Authorization": f"Bearer {access_token}"}, timeout=10.0)

        if me_resp.status_code != 200:
            from starlette.responses import HTMLResponse
            return HTMLResponse("<h3>Failed to fetch iNaturalist profile</h3>", 500)

        me = me_resp.json().get("results", [{}])[0]
        inat_login = me.get("login", "")
        display_name = me.get("name", "") or inat_login
        icon_url = me.get("icon_url", "")

        # Create session
        with _get_db() as db:
            username = f"inat:{inat_login}"
            existing = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if existing:
                user_id = existing['id']
                db.execute("UPDATE users SET display_name=?, last_seen=datetime('now') WHERE id=?",
                    (display_name, user_id))
            else:
                cur = db.execute("INSERT INTO users (username, display_name) VALUES (?,?)",
                    (username, display_name))
                user_id = cur.lastrowid

            token = _hashlib.sha256(f"inat-{inat_login}-{_time.time()}".encode()).hexdigest()[:32]
            ip = request.client.host if request.client else ''
            db.execute("INSERT INTO sessions (user_id, session_token, user_agent, ip_address) VALUES (?,?,?,?)",
                (user_id, token, request.headers.get('user-agent','')[:200], ip))
            db.execute("INSERT INTO events (user_id, event_type, details) VALUES (?, 'login', ?)",
                (user_id, _json.dumps({"provider": "inat", "inat_login": inat_login})))

        sep = '&' if '?' in redirect else '?'
        return_url = f"{redirect}{sep}auth_token={token}&auth_user={_urllib_parse.quote(display_name)}&auth_provider=iNaturalist"
        from starlette.responses import RedirectResponse
        resp = RedirectResponse(return_url, status_code=302)
        resp.delete_cookie("oauth_redirect")
        return resp


# ─── Image upload for local files (writable /tmp) ────────────────────
import uuid as _uuid
from fastapi import UploadFile, File as FastAPIFile

UPLOAD_DIR = pathlib.Path("/tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ─── Usage tracking database ─────────────────────────────────────────
import sqlite3
import time as _time
from datetime import datetime as _datetime, timedelta as _timedelta
from contextlib import contextmanager as _contextmanager

# Persistent storage: mount /data as a volume in docker-compose
USAGE_DB = pathlib.Path("/data/usage.db")
USAGE_DB.parent.mkdir(parents=True, exist_ok=True)

def _init_usage_db():
    """Create usage tracking tables if they don't exist."""
    with sqlite3.connect(str(USAGE_DB)) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                display_name TEXT,
                orcid TEXT,
                institution TEXT,
                email TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                last_seen TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(id),
                session_token TEXT UNIQUE NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                last_active TEXT DEFAULT (datetime('now')),
                user_agent TEXT,
                ip_address TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER REFERENCES sessions(id),
                user_id INTEGER REFERENCES users(id),
                event_type TEXT NOT NULL,
                details TEXT,
                timestamp TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_user ON events(user_id);
        """)
    logger.info(f"Usage DB initialized: {USAGE_DB}")

@_contextmanager
def _get_db():
    conn = sqlite3.connect(str(USAGE_DB))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

# Initialize on startup
_init_usage_db()


_ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff', '.webp'}

@app.post("/upload-image")
async def upload_image(file: UploadFile = FastAPIFile(...)):
    """Upload a local image so SAM2 can encode/decode it."""
    ext = pathlib.Path(file.filename or "img.jpg").suffix.lower()
    if ext not in _ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "Empty file")
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 50 MB)")
    uid = _uuid.uuid4().hex[:12]
    safe_name = uid + ext
    dest = UPLOAD_DIR / safe_name
    with open(dest, 'wb') as f:
        f.write(data)
    logger.info(f"Uploaded: {file.filename} -> {dest} ({len(data):,} bytes)")
    return {"url": f"/uploads/{safe_name}", "image_url": f"/uploads/{safe_name}",
            "filename": safe_name, "original_name": file.filename, "size": len(data)}

@app.get("/uploads/{filename}")
async def serve_upload(filename: str):
    """Serve uploaded images from /tmp/uploads/."""
    import re as _re
    if not _re.match(r'^[a-f0-9]{12}\.[a-z]{3,5}$', filename):
        raise HTTPException(400, "Invalid filename")
    fpath = UPLOAD_DIR / filename
    if not fpath.exists():
        raise HTTPException(404, "File not found")
    ct = {'.jpg':'image/jpeg','.jpeg':'image/jpeg','.png':'image/png',
          '.gif':'image/gif','.webp':'image/webp','.tif':'image/tiff',
          '.tiff':'image/tiff','.bmp':'image/bmp'}.get(fpath.suffix.lower(), 'application/octet-stream')
    return FileResponse(str(fpath), media_type=ct,
                        headers={"Access-Control-Allow-Origin": "*"})

# ─── Serve static HTML annotator ─────────────────────────────────────
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse as StaticFileResponse
import pathlib

STATIC_DIR = pathlib.Path("/app/static")

@app.get("/", include_in_schema=False)
async def serve_annotator():
    return StaticFileResponse(STATIC_DIR / "descriptron-gbif-annotator.html")

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


