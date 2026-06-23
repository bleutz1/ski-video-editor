"""
modal_app.py — Serverless backend for the water ski auto-reframe tool.

Wraps slalom.py and jump.py (unmodified) and exposes them via three
endpoints using a POLLING pattern (required because Modal web endpoints
have a hard 150-second timeout, but video processing takes longer):

  POST /upload          -> kicks off processing, returns {"call_id": "..."}
  GET  /status/{call_id} -> returns {"status": "running"|"done"|"error"}
  GET  /result/{call_id} -> returns the processed video bytes (once done)

Deploy with:
    modal deploy modal_app.py
"""

import modal
import subprocess
import uuid
from pathlib import Path
from fastapi import UploadFile, File, Form
from fastapi.responses import Response, JSONResponse

app = modal.App("ski-reframe")

# Container image: everything the scripts need.
# The yolov8s.pt model weights are downloaded ONCE at image-build time and
# baked into /root, so every container starts with it already present —
# no per-request download, no volume-mount complexity needed.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "opencv-python-headless",
        "ultralytics",
        "numpy",
        "fastapi[standard]",
    )
    .run_commands(
        "python -c \"from ultralytics import YOLO; YOLO('yolov8s.pt')\""
    )
    .add_local_file("slalom.py", remote_path="/root/slalom.py")
    .add_local_file("jump.py", remote_path="/root/jump.py")
)



@app.function(
    image=image,
    gpu="T4",
    timeout=600,
)
def process_video(video_bytes: bytes, mode: str, original_filename: str) -> bytes:
    work_dir = Path(f"/tmp/{uuid.uuid4()}")
    work_dir.mkdir(parents=True)

    suffix = Path(original_filename).suffix or ".mp4"
    input_path = work_dir / f"input{suffix}"
    output_path = work_dir / "output.mp4"

    input_path.write_bytes(video_bytes)

    script = "slalom.py" if mode == "slalom" else "jump.py"

    result = subprocess.run(
        ["python", f"/root/{script}", str(input_path), str(output_path)],
        capture_output=True,
        text=True,
        cwd=str(work_dir),
    )

    print("STDOUT:", result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Processing failed: {result.stderr[-2000:]}")

    final_candidates = list(work_dir.glob("output_audio.mp4"))
    final_path = final_candidates[0] if final_candidates else output_path

    if not final_path.exists():
        raise RuntimeError("Processing completed but no output file was found.")

    return final_path.read_bytes()


@app.function(image=image, timeout=90)
@modal.fastapi_endpoint(method="POST")
async def upload(file: UploadFile = File(...), mode: str = Form("slalom")):
    if mode not in ("slalom", "jump"):
        return JSONResponse({"error": "Invalid mode. Use 'slalom' or 'jump'."}, status_code=400)

    video_bytes = await file.read()
    original_filename = file.filename or "input.mp4"

    call = process_video.spawn(video_bytes, mode, original_filename)

    return JSONResponse({"call_id": call.object_id})


@app.function(image=image, timeout=60)
@modal.fastapi_endpoint(method="GET")
async def status(call_id: str):
    function_call = modal.FunctionCall.from_id(call_id)
    try:
        function_call.get(timeout=0)
        return JSONResponse({"status": "done"})
    except modal.exception.OutputExpiredError:
        return JSONResponse({"status": "error", "detail": "Result expired"}, status_code=410)
    except TimeoutError:
        return JSONResponse({"status": "running"})
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@app.function(image=image, timeout=60)
@modal.fastapi_endpoint(method="GET")
async def result(call_id: str):
    function_call = modal.FunctionCall.from_id(call_id)
    try:
        video_bytes = function_call.get(timeout=0)
    except TimeoutError:
        return JSONResponse({"error": "Not finished yet"}, status_code=425)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={"Content-Disposition": "attachment; filename=reframed.mp4"},
    )
