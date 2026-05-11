from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import quote

import cv2
import numpy as np
import psycopg2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from .extract_feature import create_landmarker, extract_single_face_features_from_bgr
    from .postgres import DB_HOST, DB_PASSWORD, DB_PORT, DB_USER, TARGET_DB
except ImportError:
    from extract_feature import create_landmarker, extract_single_face_features_from_bgr
    from postgres import DB_HOST, DB_PASSWORD, DB_PORT, DB_USER, TARGET_DB


LANDMARKER_MODEL_PATH = Path("src/models/face_landmarker.task")
FACE_SIZE = 128
IMAGES_DIR = Path("src/images_v2_face_only").resolve()

app = FastAPI(title="Face Search API")
app.mount("/static/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


class SearchRequest(BaseModel):
    vector: List[float]


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Face Search</title>
  <style>
    body { font-family: Arial, sans-serif; background:#f3f4f6; margin:0; padding:24px; }
    .card { max-width:1180px; margin:0 auto; background:#fff; border-radius:12px; padding:20px; box-shadow:0 4px 14px rgba(0,0,0,0.08); }
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
    button:disabled { background:#9ca3af; cursor:not-allowed; }
    .workspace { display:grid; grid-template-columns:minmax(0,1.35fr) minmax(340px,0.9fr); gap:16px; align-items:start; margin-top:18px; }
    .panel { min-width:0; background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:12px; }
    .panel-title { margin:0 0 10px; font-size:15px; color:#111827; }
    .preview { display:none; margin-bottom:12px; border:1px solid #e5e7eb; border-radius:8px; background:#fff; padding:10px; }
    .preview.active { display:block; }
    .preview-title { margin:0 0 8px; font-size:13px; font-weight:700; color:#111827; }
    .preview img { width:100%; max-height:260px; object-fit:contain; border-radius:6px; background:#111; display:block; }
    .preview-name { margin-top:8px; font-size:12px; color:#374151; word-break:break-word; }
    .result { font-size:14px; }
    .result pre { white-space:pre-wrap; margin:0; }
    .vector-table { width:100%; border-collapse:collapse; background:#fff; }
    .vector-table th, .vector-table td { border-bottom:1px solid #e5e7eb; padding:8px; text-align:left; vertical-align:top; }
    .vector-table th { color:#111827; font-size:12px; background:#f3f4f6; }
    .vector-table td { color:#374151; font-size:12px; }
    .vector-table .dim { width:56px; font-weight:700; color:#111827; white-space:nowrap; }
    .vector-table .value { width:92px; font-family:monospace; color:#111827; white-space:nowrap; }
    .vector-table .meaning { min-width:180px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:12px; margin-top:12px; }
    .item { border:1px solid #e5e7eb; border-radius:8px; padding:8px; background:#fff; }
    .item img { width:100%; height:130px; object-fit:cover; border-radius:6px; background:#111; }
    .meta { margin-top:6px; font-size:12px; color:#374151; word-break:break-word; }
    @media (max-width: 900px) {
      body { padding:12px; }
      .workspace { grid-template-columns:1fr; }
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Upload Face Image</h1>
    <p>System detects exactly one face, extracts the embedding first, then searches similar images on demand.</p>
    <form id="uploadForm">
      <input type="file" id="fileInput" name="file" accept="image/*" required />
      <div class="dropzone" id="dropzone">
        Drag & drop image here
      </div>
      <button type="submit">Create Embedding</button>
      <button type="button" id="searchButton" disabled>Search Similar</button>
    </form>
    <div class="workspace">
      <section class="panel">
        <h2 class="panel-title">Similar images</h2>
        <div class="result" id="searchResultBox"><pre>No search yet.</pre></div>
      </section>
      <section class="panel">
        <h2 class="panel-title">Uploaded image embedding</h2>
        <div class="preview" id="previewBox">
          <p class="preview-title">Uploaded image</p>
          <img id="previewImage" alt="Uploaded image preview" />
          <div class="preview-name" id="previewName"></div>
        </div>
        <div class="result" id="embeddingResultBox"><pre>No embedding yet.</pre></div>
      </section>
    </div>
  </div>
  <script>
    const form = document.getElementById("uploadForm");
    const searchResultBox = document.getElementById("searchResultBox");
    const embeddingResultBox = document.getElementById("embeddingResultBox");
    const fileInput = document.getElementById("fileInput");
    const dropzone = document.getElementById("dropzone");
    const searchButton = document.getElementById("searchButton");
    const previewBox = document.getElementById("previewBox");
    const previewImage = document.getElementById("previewImage");
    const previewName = document.getElementById("previewName");
    let currentEmbedding = null;
    let currentPreviewUrl = null;

    function setSelectedFile(file) {
      const dt = new DataTransfer();
      dt.items.add(file);
      fileInput.files = dt.files;
      dropzone.textContent = "Selected: " + file.name;
      currentEmbedding = null;
      searchButton.disabled = true;
      embeddingResultBox.innerHTML = "<pre>No embedding yet.</pre>";
      searchResultBox.innerHTML = "<pre>No search yet.</pre>";
      renderImagePreview(file);
    }

    function renderImagePreview(file) {
      if (currentPreviewUrl) {
        URL.revokeObjectURL(currentPreviewUrl);
      }
      currentPreviewUrl = URL.createObjectURL(file);
      previewImage.src = currentPreviewUrl;
      previewName.textContent = file.name;
      previewBox.classList.add("active");
    }

    function clearImagePreview() {
      if (currentPreviewUrl) {
        URL.revokeObjectURL(currentPreviewUrl);
      }
      currentPreviewUrl = null;
      previewImage.removeAttribute("src");
      previewName.textContent = "";
      previewBox.classList.remove("active");
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
        embeddingResultBox.innerHTML = "<pre>Please drop a valid image file.</pre>";
        return;
      }
      setSelectedFile(file);
    });

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (file) {
        dropzone.textContent = "Selected: " + file.name;
        currentEmbedding = null;
        searchButton.disabled = true;
        embeddingResultBox.innerHTML = "<pre>No embedding yet.</pre>";
        searchResultBox.innerHTML = "<pre>No search yet.</pre>";
        renderImagePreview(file);
      } else {
        dropzone.textContent = "Drag & drop image here";
        currentEmbedding = null;
        searchButton.disabled = true;
        embeddingResultBox.innerHTML = "<pre>No embedding yet.</pre>";
        searchResultBox.innerHTML = "<pre>No search yet.</pre>";
        clearImagePreview();
      }
    });

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    function renderVectorDetails(data) {
      const descriptions = data.explanation?.dimensions || [];
      const rows = (data.vector || []).map((value, index) => {
        const info = descriptions[index] || {};
        return `
          <tr>
            <td class="dim">${index + 1}</td>
            <td class="value">${Number(value).toFixed(6)}</td>
            <td class="meaning">${escapeHtml(info.represents || "")}</td>
            <td>${escapeHtml(info.calculated_by || "")}</td>
          </tr>
        `;
      }).join("");

      embeddingResultBox.innerHTML = `
        <table class="vector-table">
          <thead>
            <tr>
              <th>Chieu</th>
              <th>Gia tri</th>
              <th>Bieu dien</th>
              <th>Cach tinh</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      `;
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const file = fileInput.files[0];
      if (!file) return;
      const fd = new FormData();
      fd.append("file", file);
      currentEmbedding = null;
      searchButton.disabled = true;
      embeddingResultBox.innerHTML = "<pre>Creating embedding...</pre>";
      searchResultBox.innerHTML = "<pre>No search yet.</pre>";
      try {
        const res = await fetch("/api/embedding", { method: "POST", body: fd });
        const data = await res.json();
        if (!res.ok) {
          embeddingResultBox.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
          return;
        }

        currentEmbedding = data.vector;
        searchButton.disabled = false;
        renderVectorDetails(data);
      } catch (err) {
        embeddingResultBox.innerHTML = "<pre>Request failed: " + err + "</pre>";
      }
    });

    searchButton.addEventListener("click", async () => {
      if (!currentEmbedding) return;
      searchResultBox.innerHTML = "<pre>Searching...</pre>";
      try {
        const res = await fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ vector: currentEmbedding })
        });
        const data = await res.json();
        if (!res.ok) {
          searchResultBox.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
          return;
        }

        const top5Html = (data.top5 || []).map((item) => `
          <div class="item">
            <img src="${item.preview_url}" alt="${item.image_name}" />
            <div class="meta"><strong>${item.image_name}</strong></div>
            <div class="meta">score: ${item.score}</div>
          </div>
        `).join("");

        searchResultBox.innerHTML = `
          <pre>${JSON.stringify({
            message: data.message
          }, null, 2)}</pre>
          <div class="grid">${top5Html}</div>
        `;
      } catch (err) {
        searchResultBox.innerHTML = "<pre>Request failed: " + err + "</pre>";
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
        if vec.ndim != 1 or vec.size == 0:
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
        # fallback: serve by basename when image path is outside configured static directory
        return "/static/images/" + quote(loc.name)
    return "/static/images/" + quote(rel.as_posix(), safe="/")


def build_embedding_explanation(vector_dim: int) -> dict:
    ranges = [
        {
            "range": "1-14",
            "name": "Geometry features",
            "represents": "Khoang cach, ti le khuon mat, cac do lech landmark va hinh hoc mat-mui-mieng.",
        },
        {
            "range": "15-1790",
            "name": "HOG + gradient features",
            "represents": "Mo ta contour, huong gradient, histogram huong canh va thong ke do manh gradient.",
        },
        {
            "range": "1791-1796",
            "name": "Texture features",
            "represents": "Entropy Shannon, variance, local texture statistics, edge density va gradient mean.",
        },
        {
            "range": "1797-1808",
            "name": "Skin color features",
            "represents": "Mean/Std trong HSV va histogram kenh Hue da duoc normalize.",
        },
        {
            "range": "1809-1812",
            "name": "Symmetry features",
            "represents": "Doi xung tong the, doi xung mat, doi xung mieng va do on dinh texture hai ben.",
        },
        {
            "range": "1813-1814",
            "name": "Pose/area features",
            "represents": "Goc nghieng khuon mat (roll) va ti le dien tich khuon mat trong anh aligned.",
        },
    ]
    return {
        "note": "vector nay la embedding khuon mat theo pipeline feature da refactor (geometry + HOG/gradient + texture + color + symmetry)",
        "dimension_groups": ranges,
        "actual_dimension_count": vector_dim,
    }


async def extract_features(file: UploadFile) -> Tuple[np.ndarray, dict, Optional[JSONResponse]]:
    content = await file.read()
    arr = np.frombuffer(content, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return np.array([], dtype=np.float32), {}, JSONResponse(
            status_code=400, content={"error": "Invalid image file."}
        )

    features, metadata, status, face_count = extract_single_face_features_from_bgr(
        image_bgr=image_bgr,
        landmarker=app.state.landmarker,
        face_size=FACE_SIZE,
    )
    if status == "no_face":
        return np.array([], dtype=np.float32), {}, JSONResponse(
            status_code=400, content={"error": "No face detected in image."}
        )
    if status == "multiple_faces":
        return np.array([], dtype=np.float32), {}, JSONResponse(
            status_code=400,
            content={"error": f"Detected {face_count} faces. Please upload image with exactly 1 face."},
        )
    if status != "ok" or features is None:
        return np.array([], dtype=np.float32), {}, JSONResponse(
            status_code=400, content={"error": "Failed to extract facial features."}
        )

    return features.astype(np.float32), metadata or {}, None


@app.post("/api/embedding")
async def create_embedding(file: UploadFile = File(...)) -> JSONResponse:
    features, metadata, error_response = await extract_features(file)
    if error_response is not None:
        return error_response

    vector = [float(v) for v in features.tolist()]
    vector_dim = int(features.shape[0])
    return JSONResponse(
        content={
            "message": "ok",
            "filename": file.filename,
            "vector": vector,
            "vector_dim": vector_dim,
            "metadata": metadata,
            "explanation": build_embedding_explanation(vector_dim),
        }
    )


@app.post("/api/search")
async def search_similar(payload: SearchRequest) -> JSONResponse:
    features = np.asarray(payload.vector, dtype=np.float32)
    if features.ndim != 1 or features.size == 0:
        return JSONResponse(status_code=400, content={"error": "Vector must be a non-empty 1D array."})
    if not np.all(np.isfinite(features)):
        return JSONResponse(status_code=400, content={"error": "Vector must contain only finite numbers."})

    records = fetch_db_vectors()
    if not records:
        return JSONResponse(status_code=404, content={"error": "No image vectors found in database."})

    scored = []
    for rec in records:
        if rec["vector"].shape != features.shape:
            continue
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
    if not scored:
        return JSONResponse(
            status_code=400,
            content={"error": f"No database vectors match input dimension {int(features.shape[0])}."},
        )
    scored.sort(key=lambda x: x["score"], reverse=True)

    return JSONResponse(
        content={
            "message": "ok",
            "top5": scored[:10],
        }
    )
