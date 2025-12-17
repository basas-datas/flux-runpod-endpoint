import os
import io
import base64
import types
import torch

# =========================
# FIX для diffusers + torch
# diffusers может обращаться к torch.xpu.* при импорте,
# а в некоторых сборках torch (CUDA-only) torch.xpu отсутствует.
# =========================

def _noop(*args, **kwargs):
    return None

def _zero(*args, **kwargs):
    return 0

def _false(*args, **kwargs):
    return False

# Создаём torch.xpu, если его нет
if not hasattr(torch, "xpu"):
    torch.xpu = types.SimpleNamespace()

# Докидываем ожидаемые атрибуты, если их нет
if not hasattr(torch.xpu, "empty_cache"):
    torch.xpu.empty_cache = _noop
if not hasattr(torch.xpu, "device_count"):
    torch.xpu.device_count = _zero
if not hasattr(torch.xpu, "is_available"):
    torch.xpu.is_available = _false
if not hasattr(torch.xpu, "current_device"):
    torch.xpu.current_device = _zero
if not hasattr(torch.xpu, "synchronize"):
    torch.xpu.synchronize = _noop

# (опционально) небольшие оптимизации матмулов на NVIDIA
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

import runpod
from PIL import Image
from diffusers import FluxKontextPipeline

# =========================
# НАСТРОЙКИ
# =========================

BASE_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
LORA_REPO = "prithivMLmods/Kontext-Watermark-Remover"
LORA_FILE = "Kontext-Watermark-Remover.safetensors"

DTYPE = torch.bfloat16
DEVICE = "cuda"

# =========================
# ЗАГРУЗКА МОДЕЛИ (1 РАЗ)
# =========================

print("Loading FLUX Kontext base model...", flush=True)

pipe = FluxKontextPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=DTYPE
).to(DEVICE)

print("Loading LoRA...", flush=True)

pipe.load_lora_weights(
    LORA_REPO,
    weight_name=LORA_FILE,
    adapter_name="watermark_remover"
)

pipe.set_adapters(["watermark_remover"], adapter_weights=[1.0])

# отключаем прогресс-бар (чуть меньше мусора в логах)
try:
    pipe.set_progress_bar_config(disable=True)
except Exception:
    pass

print("Model ready.", flush=True)

# =========================
# UTILS
# =========================

def b64_to_pil(b64_str: str) -> Image.Image:
    # поддержка data-url вида: data:image/png;base64,....
    if "," in b64_str and b64_str.strip().lower().startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]

    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGB")

def pil_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# =========================
# HANDLER
# =========================

def handler(job):
    print("JOB RECEIVED", flush=True)

    inp = job.get("input", {})
    if "image_base64" not in inp:
        return {"error": "Missing 'image_base64' in input"}

    image_b64 = inp["image_base64"]
    prompt = inp.get(
        "prompt",
        "remove any watermark text or logos from the image while preserving realism"
    )

    guidance_scale = float(inp.get("guidance_scale", 2.5))
    steps = int(inp.get("steps", 28))
    seed = int(inp.get("seed", 42))

    image = b64_to_pil(image_b64)

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    with torch.inference_mode():
        result = pipe(
            image=image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            width=image.size[0],
            height=image.size[1],
            num_inference_steps=steps,
            generator=generator,
        ).images[0]

    return {"image_base64": pil_to_b64_png(result)}

runpod.serverless.start({"handler": handler})
