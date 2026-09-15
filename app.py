"""
Nobabia visual search backend.

One job only: given a product photo, return an AI "fingerprint"
(embedding vector) for it. The Nobabia WordPress site calls this once
for every product photo (to build its catalog of fingerprints) and
once for every photo a shopper uploads in the app, then does the
"which products look alike" comparison itself. This service holds no
product data at all - it only understands images.

Model: CLIP ViT-B/32 vision encoder, int8-quantized ONNX build, so it
comfortably fits Render's free 512MB web service. The weights are not
bundled in this repo - they're downloaded once from the Hugging Face
Hub the first time the service starts (or wakes up from sleep).
"""

import io
import os
import threading

import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
import onnxruntime as ort

MODEL_URL = (
    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/"
    "onnx/vision_model_quantized.onnx"
)
MODEL_PATH = "/tmp/vision_model_quantized.onnx"

# Standard CLIP preprocessing constants (openai/clip-vit-base-patch32).
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
IMAGE_SIZE = 224

app = FastAPI(title="Nobabia Image Search")

_session = None
_session_lock = threading.Lock()
_input_name = None
_output_name = None


def get_session():
    """Lazily download the model (first request only) and load it."""
    global _session, _input_name, _output_name

    if _session is not None:
        return _session

    with _session_lock:
        if _session is not None:
            return _session

        if not os.path.exists(MODEL_PATH):
            tmp_path = MODEL_PATH + ".part"
            with requests.get(MODEL_URL, timeout=180, stream=True) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            os.rename(tmp_path, MODEL_PATH)

        sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

        _input_name = sess.get_inputs()[0].name

        # Some CLIP ONNX exports return both a per-patch hidden state
        # (3 dims) and a pooled/projected embedding (2 dims). We want
        # the 2-dim one - it is what's meant to be compared between
        # images. Fall back to the first output if none matches.
        chosen = None
        for out in sess.get_outputs():
            if len(out.shape) == 2:
                chosen = out.name
                break
        _output_name = chosen or sess.get_outputs()[0].name

        _session = sess

    return _session


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    width, height = img.size
    scale = IMAGE_SIZE / min(width, height)
    new_w, new_h = round(width * scale), round(height * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    left = (new_w - IMAGE_SIZE) // 2
    top = (new_h - IMAGE_SIZE) // 2
    img = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0).astype(np.float32)

    return arr


@app.get("/")
def health():
    return {"status": "ok", "service": "nobabia-image-search"}


@app.post("/embed")
async def embed(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        pixel_values = preprocess(image_bytes)

        session = get_session()
        result = session.run([_output_name], {_input_name: pixel_values})[0]
        vector = result.reshape(-1).astype(float).tolist()

        return {"embedding": vector, "dims": len(vector)}
    except Exception as exc:  # noqa: BLE001 - surface the real reason to the caller
        raise HTTPException(status_code=500, detail=str(exc))
