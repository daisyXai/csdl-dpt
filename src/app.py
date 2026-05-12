from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import cv2
import numpy as np
import psycopg2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from .extract_feature import (
        FEATURE_VECTOR_DIM,
        create_landmarker,
        extract_single_face_features_from_bgr,
        feature_group_slice_bounds,
    )
    from .postgres import DB_HOST, DB_PASSWORD, DB_PORT, DB_USER, TARGET_DB
except ImportError:
    from extract_feature import (
        FEATURE_VECTOR_DIM,
        create_landmarker,
        extract_single_face_features_from_bgr,
        feature_group_slice_bounds,
    )
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
    .explain-note { font-size:13px; color:#374151; line-height:1.5; margin:0 0 14px; padding:10px 12px; background:#eff6ff; border-radius:8px; border:1px solid #bfdbfe; }
    .group-cards { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:16px; }
    .group-card { flex:1 1 200px; background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:10px 12px; border-left:4px solid #94a3b8; }
    .group-card.geometry { border-left-color:#2563eb; }
    .group-card.texture { border-left-color:#059669; }
    .group-card.color { border-left-color:#d97706; }
    .group-card.structural { border-left-color:#7c3aed; }
    .group-card.symmetry { border-left-color:#db2777; }
    .group-card h3 { margin:0 0 6px; font-size:13px; color:#111827; }
    .group-card .range { font-size:11px; color:#6b7280; font-family:monospace; }
    .group-card p { margin:0; font-size:12px; color:#4b5563; line-height:1.45; }
    .dim-section { margin-top:14px; }
    .dim-section h4 { margin:0 0 8px; font-size:14px; color:#111827; display:flex; align-items:center; gap:8px; }
    .dim-section h4 .badge { font-size:11px; font-weight:600; color:#fff; padding:2px 8px; border-radius:999px; }
    .badge.geometry { background:#2563eb; }
    .badge.texture { background:#059669; }
    .badge.color { background:#d97706; }
    .badge.structural { background:#7c3aed; }
    .badge.symmetry { background:#db2777; }
    .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:12px; margin-top:12px; }
    .item {
      border:1px solid #e5e7eb; border-radius:8px; padding:8px; background:#fff;
      min-width:0; max-width:100%; box-sizing:border-box; overflow:hidden;
      display:flex; flex-direction:column;
    }
    .item-thumb { width:100%; flex-shrink:0; border-radius:6px; overflow:hidden; background:#111; }
    .item-thumb img { width:100%; height:130px; object-fit:cover; display:block; }
    .item-body { margin-top:8px; min-width:0; width:100%; box-sizing:border-box; }
    .meta { font-size:12px; color:#374151; word-break:break-word; }
    .group-sim-block {
      margin-top:8px; padding:8px; background:#f8fafc; border-radius:6px; font-size:11px;
      width:100%; box-sizing:border-box; min-width:0;
    }
    .group-sim-block .sim-title { font-weight:700; color:#1e293b; margin-bottom:6px; line-height:1.3; }
    .sim-row { margin:6px 0; min-width:0; width:100%; }
    .sim-row span.lbl { display:block; font-size:11px; color:#475569; line-height:1.25; margin-bottom:4px; word-break:break-word; }
    .sim-bar-wrap { display:flex; align-items:center; gap:8px; min-width:0; width:100%; }
    .sim-bar { flex:1 1 auto; min-width:0; height:8px; background:#e2e8f0; border-radius:4px; overflow:hidden; }
    .sim-bar > i { display:block; height:100%; background:#3b82f6; border-radius:4px; }
    .sim-row span.val { flex:0 0 auto; font-family:monospace; font-size:11px; color:#0f172a; white-space:nowrap; }
    .timing-box { margin-top:12px; padding:10px 12px; background:#ecfdf5; border:1px solid #a7f3d0; border-radius:8px; font-size:13px; }
    .timing-title { margin:0 0 8px; font-weight:700; color:#065f46; font-size:12px; }
    .timing-table { width:100%; border-collapse:collapse; }
    .timing-table td { padding:4px 6px 4px 0; color:#047857; vertical-align:top; }
    .timing-table td.timing-val { text-align:right; font-family:monospace; color:#064e3b; white-space:nowrap; }
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

    const TIMING_LABELS = [
      ["feature_extraction_ms", "Trích xuất đặc trưng"],
      ["fetch_db_ms", "Truy vấn CSDL"],
      ["score_records_ms", "Tính độ tương đồng (cosine)"],
      ["sort_ms", "Sắp xếp kết quả"],
      ["search_subtotal_ms", "Tổng phần tìm kiếm"],
      ["pipeline_total_ms", "Tổng pipeline (embedding + tìm kiếm)"],
    ];

    function formatTimingMs(timingMs) {
      if (!timingMs || typeof timingMs !== "object") return "";
      const shown = new Set();
      const rows = [];
      for (const [key, label] of TIMING_LABELS) {
        if (timingMs[key] == null || timingMs[key] === undefined) continue;
        shown.add(key);
        rows.push(
          "<tr><td>" + escapeHtml(label) + '</td><td class="timing-val">' +
          Number(timingMs[key]).toFixed(3) + " ms</td></tr>"
        );
      }
      for (const key of Object.keys(timingMs).sort()) {
        if (shown.has(key)) continue;
        const v = timingMs[key];
        if (typeof v !== "number" || !Number.isFinite(v)) continue;
        rows.push(
          "<tr><td>" + escapeHtml(key) + '</td><td class="timing-val">' +
          Number(v).toFixed(3) + " ms</td></tr>"
        );
      }
      if (!rows.length) return "";
      return (
        '<div class="timing-box"><p class="timing-title">Thời gian xử lý (phía server)</p>' +
        '<table class="timing-table">' + rows.join("") + "</table></div>"
      );
    }

    const GROUP_ORDER = ["geometry", "texture", "color", "structural", "symmetry"];
    const GROUP_TITLE_VI = {
      geometry: "Hình học & tỷ lệ",
      texture: "Kết cấu vùng",
      color: "Màu HSV",
      structural: "Cạnh / gradient",
      symmetry: "Đối xứng & cường độ",
    };

    function renderVectorDetails(data) {
      const exp = data.explanation || {};
      const vec = data.vector || [];
      const dims = exp.dimensions || [];
      const groups = exp.feature_groups || [];
      const timingHtml = formatTimingMs(data.timing_ms);

      if (!dims.length || !vec.length) {
        embeddingResultBox.innerHTML =
          timingHtml +
          `<p class="explain-note">${escapeHtml(exp.note || "Không có bảng chi tiết cho kích thước vector này.")}</p>` +
          `<pre>${escapeHtml(JSON.stringify({ vector_dim: data.vector_dim, sample: vec.slice(0, 8) }, null, 2))}</pre>`;
        return;
      }

      const cardsHtml = groups
        .map(
          (g) => `
        <div class="group-card ${escapeHtml(g.key)}">
          <h3>${escapeHtml(g.title || g.key)}</h3>
          <div class="range">Chiều ${escapeHtml(g.index_range_1based)} · ${g.dim_count} số</div>
          <p>${escapeHtml(g.summary || "")}</p>
        </div>`
        )
        .join("");

      const byGroup = {};
      for (const g of GROUP_ORDER) {
        byGroup[g] = [];
      }
      for (let i = 0; i < dims.length; i++) {
        const info = dims[i] || {};
        const g = info.group || "geometry";
        if (!byGroup[g]) byGroup[g] = [];
        byGroup[g].push({ info, value: vec[i], index: i });
      }

      let sectionsHtml = "";
      for (const gkey of GROUP_ORDER) {
        const rows = (byGroup[gkey] || [])
          .map(
            ({ info, value, index }) => `
          <tr>
            <td class="dim">${info.index_1based != null ? info.index_1based : index + 1}</td>
            <td class="value">${Number(value).toFixed(6)}</td>
            <td class="meaning">${escapeHtml(info.represents || "")}</td>
            <td>${escapeHtml(info.calculated_by || "")}</td>
          </tr>`
          )
          .join("");
        const ginfo = groups.find((x) => x.key === gkey);
        const rangeTxt = ginfo ? "Chiều " + escapeHtml(ginfo.index_range_1based) : "";
        sectionsHtml += `
        <div class="dim-section">
          <h4><span class="badge ${escapeHtml(gkey)}">${escapeHtml(GROUP_TITLE_VI[gkey] || gkey)}</span> <span style="font-weight:600;color:#64748b;font-size:12px;">${rangeTxt}</span></h4>
          <table class="vector-table">
            <thead>
              <tr>
                <th>Chiều (1…52)</th>
                <th>Giá trị</th>
                <th>Ý nghĩa (vùng ảnh / đại lượng)</th>
                <th>Cách tính (tóm tắt)</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>`;
      }

      embeddingResultBox.innerHTML = `
        ${timingHtml}
        <p class="explain-note">${escapeHtml(exp.note || "")}</p>
        <p style="margin:0 0 10px;font-size:13px;color:#374151;"><strong>5 nhóm đặc trưng</strong> — mỗi hàng là một chiều sau khi ghép vector; tìm kiếm so cosine <em>toàn 52 chiều</em> và riêng <em>từng nhóm</em> (cùng đoạn chỉ số = cùng loại thông tin).</p>
        <div class="group-cards">${cardsHtml}</div>
        ${sectionsHtml}
      `;
    }

    function renderGroupSimilarityBars(simByGroup) {
      if (!simByGroup || typeof simByGroup !== "object") return "";
      const barColor = {
        geometry: "#2563eb",
        texture: "#059669",
        color: "#d97706",
        structural: "#7c3aed",
        symmetry: "#db2777",
      };
      const rows = GROUP_ORDER.map((k) => {
        const v = simByGroup[k];
        if (v === undefined || v === null) return "";
        const n = Number(v);
        const w = Math.round(Math.max(0, Math.min(1, n)) * 100);
        const col = barColor[k] || "#3b82f6";
        return `<div class="sim-row"><span class="lbl">${escapeHtml(GROUP_TITLE_VI[k] || k)}</span><div class="sim-bar-wrap"><div class="sim-bar"><i style="width:${w}%;background:${col}"></i></div><span class="val">${n.toFixed(3)}</span></div></div>`;
      }).join("");
      if (!rows) return "";
      return `<div class="group-sim-block"><div class="sim-title">Độ tương đồng theo 5 nhóm (hiển thị dưới ảnh, không phủ lên ảnh)</div>${rows}</div>`;
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
          const timing = formatTimingMs(data.timing_ms);
          embeddingResultBox.innerHTML = timing + `<pre>${JSON.stringify(data, null, 2)}</pre>`;
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
          const timing = formatTimingMs(data.timing_ms);
          searchResultBox.innerHTML = timing + `<pre>${JSON.stringify(data, null, 2)}</pre>`;
          return;
        }

        const top5Html = (data.top5 || []).map((item) => `
          <div class="item">
            <div class="item-thumb">
              <img src="${item.preview_url}" alt="${escapeHtml(item.image_name)}" />
            </div>
            <div class="item-body">
              <div class="meta"><strong>${escapeHtml(item.image_name)}</strong><br/>Toàn vector: <code>${item.score != null ? Number(item.score).toFixed(4) : ""}</code></div>
              ${renderGroupSimilarityBars(item.similarity_by_group)}
            </div>
          </div>
        `).join("");

        const searchTimingHtml = formatTimingMs(data.timing_ms);
        searchResultBox.innerHTML = `
          ${searchTimingHtml}
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


# Nhãn 5 nhóm — thứ tự ghép vector khớp extract_feature.extract_face_features_from_aligned_face.
FEATURE_GROUP_LABELS_VI: Dict[str, Dict[str, str]] = {
    "geometry": {
        "title": "Hình học & tỷ lệ (landmark)",
        "summary": "Khoảng cách và góc giữa mắt–mũi–miệng, tỷ lệ bbox mặt, độ lệch landmark.",
    },
    "texture": {
        "title": "Kết cấu theo vùng",
        "summary": "Entropy, phương sai patch mắt/mũi/miệng, độ nhám cục bộ, Laplacian toàn mặt.",
    },
    "color": {
        "title": "Màu HSV",
        "summary": "Histogram 3 bin trên kênh H/S/V và giá trị trung bình H/S/V (đã chuẩn hóa 0–1).",
    },
    "structural": {
        "title": "Cấu trúc cạnh / gradient",
        "summary": "Sobel trên patch mặt, mật độ Canny theo vùng, thống kê contour.",
    },
    "symmetry": {
        "title": "Đối xứng & cường độ",
        "summary": "Điểm đối xứng patch mặt/mắt/miệng; trung bình, phương sai, độ tương phản cường độ.",
    },
}

# 52 dòng: (group_key, mô tả ý nghĩa, cách tính ngắn) — chỉ số 1-based hiển thị theo thứ tự ghép vector.
_VECTOR_52_ROWS: List[Tuple[str, str, str]] = [
    ("geometry", "Khoảng cách hai mắt so với chiều ngang vùng mặt", "‖mắt trái − mắt phải‖ / face_w"),
    ("geometry", "Mắt trái đến mũi (theo chiều cao mặt)", "‖mắt trái − mũi‖ / face_h"),
    ("geometry", "Mắt phải đến mũi", "‖mắt phải − mũi‖ / face_h"),
    ("geometry", "Mũi đến tâm miệng", "‖mũi − tâm miệng‖ / face_h"),
    ("geometry", "Độ rộng miệng so với ngang mặt", "‖khóe trái − khóe phải‖ / face_w"),
    ("geometry", "Tỷ lệ cao/rộng khuôn mặt (đã scale)", "(face_h / face_w) / 2, clamp 0–1"),
    ("geometry", "Tỷ lệ vùng trán (trên đường mắt)", "(eye_line_y − fy1) / face_h"),
    ("geometry", "Tỷ lệ hàm (miệng so khoảng cách mắt)", "mouth_width / eye_dist (chuẩn hóa)"),
    ("geometry", "Tỷ lệ vùng dưới mũi (cằm)", "(fy2 − mũi_y) / face_h"),
    ("geometry", "Lệch dọc giữa hai mắt", "|y_mắt trái − y_mắt phải| / face_h"),
    ("geometry", "Lệch ngang mũi so tâm mặt", "|mũi_x − face_center_x| / face_w"),
    ("geometry", "Lệch ngang miệng so tâm mặt", "|tâm miệng_x − face_center_x| / face_w"),
    ("geometry", "Góc tam giác mắt trái–mũi–mắt phải", "∠(mắt trái, mũi, mắt phải) / 180°"),
    ("geometry", "Góc khóe trái–mũi–khóe phải", "∠(miệng trái, mũi, miệng phải) / 180°"),
    ("texture", "Entropy patch vùng mắt", "Shannon trên patch mắt (Gaussian blur nhẹ)"),
    ("texture", "Entropy patch vùng mũi", "Shannon trên patch mũi"),
    ("texture", "Entropy patch vùng miệng", "Shannon trên patch miệng"),
    ("texture", "Entropy toàn ảnh xám mặt", "Shannon trên toàn face gray"),
    ("texture", "Phương sai cường độ patch mắt", "var(patch) / 255²"),
    ("texture", "Phương sai patch mũi", "var(patch) / 255²"),
    ("texture", "Phương sai patch miệng", "var(patch) / 255²"),
    ("texture", "Độ lệch chuẩn cục bộ trung bình", "mean(local_std) trên ảnh, /255"),
    ("texture", "Độ phân tán của local_std", "std(local_std) / 255"),
    ("texture", "Độ “nét” Laplacian toàn mặt", "mean(|Laplacian|) / 255"),
    ("color", "Histogram Hue bin 1 (3 bin)", "hist H [0,180), normalized"),
    ("color", "Histogram Hue bin 2", "hist H normalized"),
    ("color", "Histogram Hue bin 3", "hist H normalized"),
    ("color", "Histogram Saturation bin 1", "hist S [0,256), normalized"),
    ("color", "Histogram Saturation bin 2", "hist S normalized"),
    ("color", "Histogram Saturation bin 3", "hist S normalized"),
    ("color", "Histogram Value bin 1", "hist V normalized"),
    ("color", "Histogram Value bin 2", "hist V normalized"),
    ("color", "Histogram Value bin 3", "hist V normalized"),
    ("color", "Sắc độ trung bình (Hue)", "mean(H channel) / 179"),
    ("color", "Độ bão hòa trung bình", "mean(S) / 255"),
    ("color", "Độ sáng trung bình", "mean(V) / 255"),
    ("structural", "Gradient Sobel — độ lớn trung bình", "mean(magnitude) / 255 trên patch mặt"),
    ("structural", "Gradient Sobel — độ lệch độ lớn", "std(magnitude) / 255"),
    ("structural", "Sobel X — trung bình |Gx|", "mean(|sobel_x|) / 255"),
    ("structural", "Sobel Y — trung bình |Gy|", "mean(|sobel_y|) / 255"),
    ("structural", "Mật độ cạnh Canny — patch mắt", "tỷ lệ pixel cạnh / diện tích patch"),
    ("structural", "Mật độ cạnh — patch mũi", "Canny trên patch mũi"),
    ("structural", "Mật độ cạnh — patch miệng", "Canny patch miệng"),
    ("structural", "Mật độ cạnh — toàn patch mặt", "Canny toàn vùng mặt"),
    ("structural", "Contour — độ dài cung trung bình", "mean(arcLength) / 256"),
    ("structural", "Contour — diện tích trung bình (chuẩn hóa patch)", "mean(area) / (patch w×h)"),
    ("symmetry", "Đối xứng gương toàn mặt", "1 − mean|trái − flip(phải)| / 255"),
    ("symmetry", "Đối xứng vùng hai mắt", "so sánh patch mắt trái vs flip(mắt phải)"),
    ("symmetry", "Đối xứng patch miệng", "symmetry_score trên patch miệng"),
    ("symmetry", "Cường độ trung bình (patch mặt)", "mean(gray) / 255"),
    ("symmetry", "Phương sai cường độ patch mặt", "var(gray) / 255²"),
    ("symmetry", "Độ tương phản (std cường độ)", "std(gray) / 128"),
]


def _build_52_dimension_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for i, (gkey, rep, calc) in enumerate(_VECTOR_52_ROWS):
        rows.append(
            {
                "index_1based": i + 1,
                "group": gkey,
                "group_title": FEATURE_GROUP_LABELS_VI[gkey]["title"],
                "represents": rep,
                "calculated_by": calc,
            }
        )
    return rows


def build_embedding_explanation(vector_dim: int) -> dict:
    groups_meta = []
    for gkey, start, end in feature_group_slice_bounds():
        lab = FEATURE_GROUP_LABELS_VI[gkey]
        groups_meta.append(
            {
                "key": gkey,
                "index_range_1based": f"{start + 1}-{end}",
                "dim_count": end - start,
                "title": lab["title"],
                "summary": lab["summary"],
            }
        )
    if vector_dim == FEATURE_VECTOR_DIM and len(_VECTOR_52_ROWS) == FEATURE_VECTOR_DIM:
        return {
            "note": (
                "Vector 52 chiều: ghép 5 nhóm đặc trưng thủ công (geometry → texture → color → "
                "structural → symmetry), giá trị đã clip [0,1]. So sánh tìm kiếm có thêm cosine "
                "theo từng nhóm (cùng đoạn chỉ số vector = cùng loại thông tin trên ảnh)."
            ),
            "expected_dim": FEATURE_VECTOR_DIM,
            "feature_groups": groups_meta,
            "dimensions": _build_52_dimension_rows(),
        }
    return {
        "note": (
            f"Kích thước vector hiện tại là {vector_dim}; giải thích chi tiết 52 chiều chỉ áp dụng "
            f"khi vector = {FEATURE_VECTOR_DIM} (pipeline extract_feature v3)."
        ),
        "expected_dim": FEATURE_VECTOR_DIM,
        "actual_dimension_count": vector_dim,
        "feature_groups": groups_meta if vector_dim == FEATURE_VECTOR_DIM else [],
        "dimensions": [],
    }


def cosine_similarity_per_group(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Cosine giữa hai vector cùng cắt theo từng nhóm 5 vùng (cùng chiều con)."""
    out: Dict[str, float] = {}
    for gkey, start, end in feature_group_slice_bounds():
        out[gkey] = cosine_similarity(a[start:end], b[start:end])
    return out


def _ms_since(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 3)


async def extract_features(
    file: UploadFile,
) -> Tuple[np.ndarray, dict, Optional[JSONResponse], Dict[str, float]]:
    timing: Dict[str, float] = {"feature_extraction_ms": 0.0}
    content = await file.read()
    arr = np.frombuffer(content, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return np.array([], dtype=np.float32), {}, JSONResponse(
            status_code=400, content={"error": "Invalid image file.", "timing_ms": timing}
        ), timing

    t_extract = time.perf_counter()
    features, metadata, status, face_count = extract_single_face_features_from_bgr(
        image_bgr=image_bgr,
        landmarker=app.state.landmarker,
        face_size=FACE_SIZE,
    )
    timing["feature_extraction_ms"] = _ms_since(t_extract)

    if status == "no_face":
        return np.array([], dtype=np.float32), {}, JSONResponse(
            status_code=400,
            content={
                "error": "No face detected in image.",
                "timing_ms": timing,
            },
        ), timing
    if status == "multiple_faces":
        return np.array([], dtype=np.float32), {}, JSONResponse(
            status_code=400,
            content={
                "error": f"Detected {face_count} faces. Please upload image with exactly 1 face.",
                "timing_ms": timing,
            },
        ), timing
    if status != "ok" or features is None:
        return np.array([], dtype=np.float32), {}, JSONResponse(
            status_code=400,
            content={
                "error": "Failed to extract facial features.",
                "timing_ms": timing,
            },
        ), timing

    return features.astype(np.float32), metadata or {}, None, timing


def search_similarity_core(
    features: np.ndarray,
) -> Tuple[List[dict], Dict[str, float], Optional[str]]:
    """
    Nạp vector từ DB, tính cosine với query, sắp xếp giảm dần theo score.
    Trả về (danh_sách_đã_chấm_điểm, timing_ms, lỗi_ngắn hoặc None).
    """
    timings: Dict[str, float] = {
        "fetch_db_ms": 0.0,
        "score_records_ms": 0.0,
        "sort_ms": 0.0,
    }
    t0 = time.perf_counter()
    records = fetch_db_vectors()
    timings["fetch_db_ms"] = _ms_since(t0)
    if not records:
        return [], timings, "no_vectors"

    t1 = time.perf_counter()
    scored: List[dict] = []
    for rec in records:
        if rec["vector"].shape != features.shape:
            continue
        sim = cosine_similarity(features, rec["vector"])
        group_sim = cosine_similarity_per_group(features, rec["vector"])
        scored.append(
            {
                "id": rec["id"],
                "image_name": rec["image_name"],
                "image_location": rec["image_location"],
                "preview_url": build_preview_url(rec["image_location"]),
                "score": round(sim, 6),
                "similarity_by_group": {k: round(float(v), 6) for k, v in group_sim.items()},
            }
        )
    timings["score_records_ms"] = _ms_since(t1)

    if not scored:
        return [], timings, "dimension_mismatch"

    t2 = time.perf_counter()
    scored.sort(key=lambda x: x["score"], reverse=True)
    timings["sort_ms"] = _ms_since(t2)

    timings["search_subtotal_ms"] = round(
        timings["fetch_db_ms"] + timings["score_records_ms"] + timings["sort_ms"], 3
    )
    return scored, timings, None


@app.post("/api/embedding")
async def create_embedding(file: UploadFile = File(...)) -> JSONResponse:
    features, metadata, error_response, timing = await extract_features(file)
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
            "timing_ms": timing,
        }
    )


@app.post("/api/embedding_and_search")
async def embedding_and_search(file: UploadFile = File(...)) -> JSONResponse:
    """
    Một luồng: trích đặc trưng rồi tìm ảnh tương tự; trả về timing cho từng đoạn.
    """
    features, metadata, error_response, embed_timing = await extract_features(file)
    if error_response is not None:
        return error_response

    scored, search_timing, err = search_similarity_core(features)
    timing_ms: Dict[str, Any] = {
        **embed_timing,
        **search_timing,
        "pipeline_total_ms": round(
            embed_timing.get("feature_extraction_ms", 0.0)
            + search_timing.get("search_subtotal_ms", 0.0),
            3,
        ),
    }

    if err == "no_vectors":
        return JSONResponse(
            status_code=404,
            content={
                "error": "No image vectors found in database.",
                "timing_ms": timing_ms,
            },
        )
    if err == "dimension_mismatch":
        return JSONResponse(
            status_code=400,
            content={
                "error": f"No database vectors match input dimension {int(features.shape[0])}.",
                "timing_ms": timing_ms,
            },
        )

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
            "top5": scored[:10],
            "timing_ms": timing_ms,
        }
    )


@app.post("/api/search")
async def search_similar(payload: SearchRequest) -> JSONResponse:
    features = np.asarray(payload.vector, dtype=np.float32)
    if features.ndim != 1 or features.size == 0:
        return JSONResponse(status_code=400, content={"error": "Vector must be a non-empty 1D array."})
    if not np.all(np.isfinite(features)):
        return JSONResponse(status_code=400, content={"error": "Vector must contain only finite numbers."})

    scored, timing, err = search_similarity_core(features)

    if err == "no_vectors":
        return JSONResponse(
            status_code=404,
            content={"error": "No image vectors found in database.", "timing_ms": timing},
        )
    if err == "dimension_mismatch":
        return JSONResponse(
            status_code=400,
            content={
                "error": f"No database vectors match input dimension {int(features.shape[0])}.",
                "timing_ms": timing,
            },
        )

    return JSONResponse(
        content={
            "message": "ok",
            "top5": scored[:5],
            "timing_ms": timing,
        }
    )
