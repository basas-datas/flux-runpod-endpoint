import base64
import io
import os
import random
from threading import Lock

import runpod
import torch
import numpy as np
from PIL import Image, ImageOps
from diffusers import FluxKontextPipeline

# ------------------------------------------------------------------
# STORAGE CONFIG
# ------------------------------------------------------------------

HF_HOME = "/runpod-volume/hf"
HF_HUB_CACHE = f"{HF_HOME}/hub"
TRANSFORMERS_CACHE = f"{HF_HOME}/transformers"

os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

os.makedirs(HF_HUB_CACHE, exist_ok=True)
os.makedirs(TRANSFORMERS_CACHE, exist_ok=True)

# ------------------------------------------------------------------
# MODEL CONFIG
# ------------------------------------------------------------------

HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SIDE = 1024
MAX_SEED = 2**31 - 1
PAD_MULTIPLE = 16  # padding to multiple-of-16 then crop back

# Prefer BF16 on CUDA if supported, else fallback
if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
    DTYPE = torch.bfloat16
else:
    # CPU -> float32; CUDA without bf16 -> float16
    DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16

# Optional: helps performance/stability on Ampere+
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True

PIPE_LOCK = Lock()

# ------------------------------------------------------------------
# MODEL INIT
# ------------------------------------------------------------------

pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=DTYPE,
    token=HF_TOKEN,
).to(DEVICE)

pipe.load_lora_weights(
    "prithivMLmods/Kontext-Watermark-Remover",
    weight_name="Kontext-Watermark-Remover.safetensors",
    adapter_name="watermark_remover",
)
pipe.set_adapters(["watermark_remover"], adapter_weights=[1.0])

# If you want extra robustness, you can keep VAE in fp32:
# (sometimes reduces NaNs/black outputs)
try:
    pipe.vae.to(dtype=torch.float32)
except Exception:
    pass

# ------------------------------------------------------------------
# IMAGE HELPERS
# ------------------------------------------------------------------

def base64_to_pil(base64_str: str) -> Image.Image:
    # supports data URLs like "data:image/png;base64,...."
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    raw = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)  # respect EXIF orientation
    img.load()
    return img

def remove_alpha_force_rgb(img: Image.Image) -> Image.Image:
    """
    Ensures output is RGB (no alpha). If input has transparency,
    composite over a background color estimated from border pixels;
    fallback to white.
    """
    # Normalize palette images with transparency etc.
    if img.mode in ("P", "LA"):
        img = img.convert("RGBA")

    if img.mode == "RGBA":
        arr = np.array(img, dtype=np.uint8)
        rgb = arr[..., :3].astype(np.float32)
        a = arr[..., 3].astype(np.float32) / 255.0

        # estimate background from border pixels where alpha is present
        h, w = a.shape
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        mask = border & (a > 0.1)
        if mask.any():
            bg = np.median(rgb[mask], axis=0)
        else:
            bg = np.array([255.0, 255.0, 255.0], dtype=np.float32)

        comp = rgb * a[..., None] + bg[None, None, :] * (1.0 - a[..., None])
        out = np.clip(comp, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

    # CMYK / L / etc -> RGB
    return img.convert("RGB")

def resize_long_side(img: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return img.resize((new_w, new_h), Image.LANCZOS)
    return img

def pad_to_multiple_edge(img: Image.Image, multiple: int = 16):
    """
    Pads RGB image to (H,W) multiples using edge replication.
    Returns padded_image, crop_box_to_restore (left, top, right, bottom)
    """
    w, h = img.size
    pad_w = (multiple - (w % multiple)) % multiple
    pad_h = (multiple - (h % multiple)) % multiple

    if pad_w == 0 and pad_h == 0:
        return img, (0, 0, w, h)

    arr = np.array(img, dtype=np.uint8)
    # pad right and bottom with edge values
    arr_padded = np.pad(
        arr,
        pad_width=((0, pad_h), (0, pad_w), (0, 0)),
        mode="edge",
    )
    padded = Image.fromarray(arr_padded, mode="RGB")
    # crop box to restore original resized dims
    return padded, (0, 0, w, h)

def pil_to_base64_png_rgb(img: Image.Image) -> str:
    # ensure no alpha in output
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ------------------------------------------------------------------
# HANDLER
# ------------------------------------------------------------------

def handler(job):
    job_input = job.get("input", {})

    if "image" not in job_input:
        return {"error": "image (base64) is required"}

    prompt = job_input.get(
        "prompt",
        "[photo content], remove any watermark, text or logos"
    )

    guidance_scale = float(job_input.get("guidance_scale", 2.5))
    steps = int(job_input.get("steps", 28))

    seed = int(job_input.get("seed", 42))
    randomize_seed = bool(job_input.get("randomize_seed", False))
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # ---- Load & normalize image (any format) ----
    img = base64_to_pil(job_input["image"])
    img = remove_alpha_force_rgb(img)        # обязательное удаление alpha
    img = resize_long_side(img, MAX_SIDE)    # сохранить пропорцию, max side <= 1024

    # Pad to stable sizes (multiple-of-16), then crop back after
    img_padded, crop_box = pad_to_multiple_edge(img, PAD_MULTIPLE)

    generator = None
    if DEVICE == "cuda":
        generator = torch.Generator(device="cuda").manual_seed(seed)

    # ---- Inference (lock for thread safety) ----
    with PIPE_LOCK:
        if DEVICE == "cuda" and DTYPE == torch.bfloat16:
            autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        elif DEVICE == "cuda" and DTYPE == torch.float16:
            autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
        else:
            # CPU: no autocast
            class _Noop:
                def __enter__(self): return None
                def __exit__(self, *args): return False
            autocast_ctx = _Noop()

        with torch.inference_mode(), autocast_ctx:
            out = pipe(
                image=img_padded,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                width=img_padded.width,
                height=img_padded.height,
                generator=generator,
            ).images[0]

    # Crop back to resized original dims (preserve proportion)
    out = out.crop(crop_box)

    # Ensure RGB output (no alpha) and PNG base64
    return {
        "image": pil_to_base64_png_rgb(out),
        "seed": seed,
        "width": out.width,
        "height": out.height,
    }

# ------------------------------------------------------------------
# START SERVERLESS
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting FLUX Kontext Serverless Worker (bf16 preferred)")
    runpod.serverless.start({"handler": handler})
