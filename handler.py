import os
import io
import base64
import types
import torch

# =========================
# FIX для diffusers + torch
# Исправляет AttributeError: 'types.SimpleNamespace' object has no attribute 'device_count'
# =========================
if not hasattr(torch, "xpu"):
    torch.xpu = types.SimpleNamespace(
        empty_cache=lambda: None,
        device_count=lambda: 0,
        is_available=lambda: False
    )

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
# Используем device_map="auto" или явный .to(DEVICE)
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
pipe.enable_model_cpu_offload = False

print("Model ready.", flush=True)

# =========================
# UTILS
# =========================

def b64_to_pil(b64):
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")

def pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

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

    try:
        image = b64_to_pil(image_b64)
        
        # Генератор для воспроизводимости
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        print(f"Running inference with prompt: {prompt}", flush=True)
        
        result = pipe(
            image=image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            width=image.size[0],
            height=image.size[1],
            num_inference_steps=steps,
            generator=generator,
        ).images[0]

        return {
            "image_base64": pil_to_b64(result)
        }
    except Exception as e:
        print(f"Error during inference: {str(e)}", flush=True)
        return {"error": str(e)}

# Запуск сервера
runpod.serverless.start({"handler": handler})