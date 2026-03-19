"""Custom HTML/CSS/JS UI for the noise-to-meaning diffusion demo."""

from __future__ import annotations

import json
import mimetypes
import os
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import torch

from diffusion_core import (
    DEFAULT_SEED,
    DEFAULT_STEPS,
    MAX_STEPS,
    MIN_STEPS,
    capture_progression,
    condition_prompt,
    load_pipeline,
    frame_label,
    save_progression_frames,
    PROMPT_MODE_DIRECT,
)

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
OUTPUTS_DIR = ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "ui_runs"
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
DEFAULT_MODE = os.environ.get("DEFAULT_MODE", PROMPT_MODE_DIRECT)
generation_lock = threading.Lock()

DEVICE = os.environ.get("DEVICE")
if not DEVICE:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

UI_MAX_FRAMES_RAW = os.environ.get("UI_MAX_FRAMES")
if UI_MAX_FRAMES_RAW is None:
    # On GPU, default to showing *all* frames; on CPU keep the old cap.
    UI_MAX_FRAMES = None if DEVICE.startswith("cuda") else 8
else:
    val = UI_MAX_FRAMES_RAW.strip().lower()
    if val in {"", "all", "none"}:
        UI_MAX_FRAMES = None
    else:
        UI_MAX_FRAMES = int(val)

WARMUP_MODEL = os.environ.get("WARMUP_MODEL", "1").lower() in {"1", "true", "yes", "on"}


def json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")


def safe_resolve(base: Path, request_path: str) -> Path | None:
    cleaned = unquote(request_path).lstrip("/")
    candidate = (base / cleaned).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate


def write_response(handler: BaseHTTPRequestHandler, status: int, content: bytes, content_type: str) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(content)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    try:
        handler.wfile.write(content)
    except (BrokenPipeError, ConnectionResetError):
        return


def write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    write_response(handler, status, json_bytes(payload), "application/json; charset=utf-8")


def serve_file(handler: BaseHTTPRequestHandler, path: Path) -> None:
    if not path.is_file():
        write_json(handler, HTTPStatus.NOT_FOUND, {"error": "Not found"})
        return

    content_type, _ = mimetypes.guess_type(path.name)
    content = path.read_bytes()
    write_response(handler, HTTPStatus.OK, content, content_type or "application/octet-stream")


def run_generation(seed_text: str, steps: int, mode: str) -> dict:
    if not MIN_STEPS <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be between {MIN_STEPS} and {MAX_STEPS}")

    conditioned = condition_prompt(seed_text, mode=mode)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    frames = capture_progression(
        conditioned.resolved_prompt,
        steps,
        DEFAULT_SEED,
        max_frames=UI_MAX_FRAMES,
        device=DEVICE,
    )
    assets = save_progression_frames(frames, run_dir)
    elapsed_seconds = round(time.perf_counter() - start, 2)

    base_url = f"/outputs/ui_runs/{run_id}"
    frame_urls = [f"{base_url}/frames/{path.name}" for path in assets.frame_paths]

    return {
        "run_id": run_id,
        "mode": conditioned.mode,
        "mode_label": conditioned.mode_label,
        "seed_text": conditioned.seed_text,
        "input_label": conditioned.input_label,
        "completion_caption": conditioned.completion_caption,
        "resolved_prompt": conditioned.resolved_prompt,
        "prompt": conditioned.resolved_prompt,
        "steps": steps,
        "seed": DEFAULT_SEED,
        "elapsed_seconds": elapsed_seconds,
        "frame_count": len(frame_urls),
        "frames": [
            {"index": index, "label": frame_label(index), "url": url}
            for index, url in enumerate(frame_urls)
        ],
        "final_image_url": f"{base_url}/final_image.png",
    }


class NoiseMeaningHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            serve_file(self, WEB_DIR / "index.html")
            return

        if path.startswith("/outputs/"):
            target = safe_resolve(OUTPUTS_DIR, path[len("/outputs/") :])
            if target is None:
                write_json(self, HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            serve_file(self, target)
            return

        target = safe_resolve(WEB_DIR, path.lstrip("/"))
        if target is None:
            write_json(self, HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        serve_file(self, target)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/generate":
            write_json(self, HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length) or b"{}")
        steps = int(payload.get("steps", DEFAULT_STEPS))
        mode = str(payload.get("mode") or DEFAULT_MODE or PROMPT_MODE_DIRECT)
        seed_text = str(payload.get("seed_text") or payload.get("prompt") or "")

        if not generation_lock.acquire(blocking=False):
            write_json(self, HTTPStatus.CONFLICT, {"error": "Generation already in progress"})
            return

        try:
            result = run_generation(seed_text, steps, mode)
            write_json(self, HTTPStatus.OK, result)
        except ValueError as exc:
            write_json(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - surfaced to the browser
            traceback.print_exc()
            write_json(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
        finally:
            generation_lock.release()


def main() -> None:
    if WARMUP_MODEL:
        print("Warming up diffusion pipeline...", flush=True)
        # This shifts the "model load" cost away from the first /api/generate response.
        load_pipeline(DEVICE)
    server = ThreadingHTTPServer((HOST, PORT), NoiseMeaningHandler)
    print(f"Serving on http://{HOST}:{PORT}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
