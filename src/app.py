from __future__ import annotations

import json
from pathlib import Path
from typing import List
from urllib.parse import quote

import cv2
import numpy as np
import psycopg2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from .extract_feature import create_landmarker, extract_single_face_features_from_bgr
    from .postgres import DB_HOST, DB_PASSWORD, DB_PORT, DB_USER, TARGET_DB
except ImportError:
    from extract_feature import create_landmarker, extract_single_face_features_from_bgr
    from postgres import DB_HOST, DB_PASSWORD, DB_PORT, DB_USER, TARGET_DB


LANDMARKER_MODEL_PATH = Path("src/models/face_landmarker.task")
FACE_SIZE = 128
IMAGES_DIR = Path("src/images").resolve()

app = FastAPI(title="Face Search API")
app.mount("/static/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Face Search</title>
  <style>
    body { font-family: Arial, sans-serif; background:#f3f4f6; margin:0; padding:24px; }
    .card { max-width:700px; margin:0 auto; background:#fff; border-radius:12px; padding:20px; box-shadow:0 4px 14px rgba(0,0,0,0.08); }
    h1 { margin-top:0; font-size:24px; }
    input[type=file] { display:block; margin:12px 0; }
    .dropzone {
      margin: 12px 0;
      padding: 20px;
      border: 2px dashed #9ca3af;
      border-radius: 10px;
      background: #f8fafc;
      color: #374151;
      text-align: center;
      transition: all 0.2s ease;
    }
    .dropzone.dragover {
      border-color: #2563eb;
      background: #eff6ff;
      color: #1d4ed8;
    }
    button { background:#2563eb; color:#fff; border:none; padding:10px 14px; border-radius:8px; cursor:pointer; }
    button:hover { background:#1d4ed8; }
    .result { margin-top:18px; background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:12px; font-size:14px; }
    .result pre { white-space:pre-wrap; margin:0; }
    .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:12px; margin-top:12px; }
    .item { border:1px solid #e5e7eb; border-radius:8px; padding:8px; background:#fff; }
    .item img { width:100%; height:130px; object-fit:cover; border-radius:6px; background:#111; }
    .meta { margin-top:6px; font-size:12px; color:#374151; word-break:break-word; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Upload Face Image</h1>
    <p>System detects exactly one face, extracts 30 features, and finds top-5 similar images.</p>
    <form id="uploadForm">
      <input type="file" id="fileInput" name="file" accept="image/*" required />
      <div class="dropzone" id="dropzone">
        Drag & drop image here
      </div>
      <button type="submit">Search Similar</button>
    </form>
    <div class="result" id="resultBox"><pre>No result yet.</pre></div>
  </div>
  <script>
    const form = document.getElementById("uploadForm");
    const resultBox = document.getElementById("resultBox");
    const fileInput = document.getElementById("fileInput");
    const dropzone = document.getElementById("dropzone");

    function setSelectedFile(file) {
      const dt = new DataTransfer();
      dt.items.add(file);
      fileInput.files = dt.files;
      dropzone.textContent = "Selected: " + file.name;
    }

    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", (e) => {
      const files = e.dataTransfer.files;
      if (!files || files.length === 0) return;
      const file = files[0];
      if (!file.type.startsWith("image/")) {
        resultBox.textContent = "Please drop a valid image file.";
        return;
      }
      setSelectedFile(file);
    });

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (file) {
        dropzone.textContent = "Selected: " + file.name;
      } else {
        dropzone.textContent = "Drag & drop image here";
      }
    });

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const file = fileInput.files[0];
      if (!file) return;
      const fd = new FormData();
      fd.append("file", file);
      resultBox.innerHTML = "<pre>Processing...</pre>";
      try {
        const res = await fetch("/api/search", { method: "POST", body: fd });
        const data = await res.json();
        if (!res.ok) {
          resultBox.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
          return;
        }

        const top5Html = (data.top5 || []).map((item) => `
          <div class="item">
            <img src="${item.preview_url}" alt="${item.image_name}" />
            <div class="meta"><strong>${item.image_name}</strong></div>
            <div class="meta">score: ${item.score}</div>
          </div>
        `).join("");

        resultBox.innerHTML = `
          <pre>${JSON.stringify({
            message: data.message,
            uploaded_file: data.uploaded_file,
            uploaded_vector_dim: data.uploaded_vector_dim
          }, null, 2)}</pre>
          <div class="grid">${top5Html}</div>
        `;
      } catch (err) {
        resultBox.innerHTML = "<pre>Request failed: " + err + "</pre>";
      }
    });
  </script>
</body>
</html>
"""


@app.on_event("startup")
def startup_event() -> None:
    # num_faces=5 to explicitly reject multi-face uploads.
    app.state.landmarker = create_landmarker(LANDMARKER_MODEL_PATH, num_faces=5)


@app.on_event("shutdown")
def shutdown_event() -> None:
    if hasattr(app.state, "landmarker") and app.state.landmarker is not None:
        app.state.landmarker.close()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def parse_vector_literal(vector_literal: str) -> np.ndarray:
    # pgvector text literal looks like: [0.1,0.2,...]
    values = np.fromstring(vector_literal.strip("[]"), sep=",", dtype=np.float32)
    return values


def fetch_db_vectors() -> List[dict]:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=TARGET_DB,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, image_name, image_location, image_metadata::text, vector::text
                FROM images
                WHERE vector IS NOT NULL;
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    data = []
    for row in rows:
        vec = parse_vector_literal(row[4])
        if vec.shape != (30,):
            continue
        data.append(
            {
                "id": row[0],
                "image_name": row[1],
                "image_location": row[2],
                "image_metadata": json.loads(row[3]) if row[3] else {},
                "vector": vec,
            }
        )
    return data


def build_preview_url(image_location: str) -> str:
    loc = Path(image_location).resolve()
    try:
        rel = loc.relative_to(IMAGES_DIR)
    except ValueError:
        # fallback: send absolute path encoded (not expected if data is under src/images)
        return "/static/images/" + quote(loc.name)
    return "/static/images/" + quote(rel.as_posix(), safe="/")


@app.post("/api/search")
async def search_similar(file: UploadFile = File(...)) -> JSONResponse:
    content = await file.read()
    arr = np.frombuffer(content, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})

    features, metadata, status, face_count = extract_single_face_features_from_bgr(
        image_bgr=image_bgr,
        landmarker=app.state.landmarker,
        face_size=FACE_SIZE,
    )
    if status == "no_face":
        return JSONResponse(status_code=400, content={"error": "No face detected in image."})
    if status == "multiple_faces":
        return JSONResponse(
            status_code=400,
            content={"error": f"Detected {face_count} faces. Please upload image with exactly 1 face."},
        )
    if status != "ok" or features is None:
        return JSONResponse(status_code=400, content={"error": "Failed to extract facial features."})

    records = fetch_db_vectors()
    if not records:
        return JSONResponse(status_code=404, content={"error": "No image vectors found in database."})

    scored = []
    for rec in records:
        sim = cosine_similarity(features, rec["vector"])
        scored.append(
            {
                "id": rec["id"],
                "image_name": rec["image_name"],
                "image_location": rec["image_location"],
                "preview_url": build_preview_url(rec["image_location"]),
                "score": round(sim, 6),
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)

    return JSONResponse(
        content={
            "message": "ok",
            "uploaded_file": file.filename,
            "uploaded_face_metadata": metadata,
            "uploaded_vector_dim": int(features.shape[0]),
            "top5": scored[:100],
        }
    )
