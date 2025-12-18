import base64
import io
import os
import runpod
import torch
from PIL import Image
from diffusers import FluxKontextPipeline

# ------------------------------------------------------------------
# STORAGE CONFIG (КАК В COMFYUI)
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
DTYPE = torch.float16
MAX_SIDE = 1024

# ------------------------------------------------------------------
# MODEL INIT (ONCE PER WORKER)
# ------------------------------------------------------------------

pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=DTYPE,
    token=HF_TOKEN,
).to(DEVICE)

pipe.load_lora_weights(
    "prithivMLmods/Kontext-Watermark-Remover",
    weight_name="Kontext-Watermark-Remover.safetensors",
    adapter_name="watermark_remover"
)

pipe.set_adapters(["watermark_remover"], adapter_weights=[1.0])

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def resize_if_needed(image: Image.Image) -> Image.Image:
    w, h = image.size
    scale = min(MAX_SIDE / max(w, h), 1.0)
    if scale < 1.0:
        return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image

def base64_to_pil(base64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(base64_str))).convert("RGB")

def pil_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ------------------------------------------------------------------
# HANDLER
# ------------------------------------------------------------------

def handler(job):
    job_input = job["input"]

    if "image" not in job_input:
        return {"error": "image (base64) is required"}

    prompt = job_input.get(
        "prompt",
        "[photo content], remove subtitles, captions, timestamps, watermarks, or any text overlays. Reconstruct the background naturally and preserve realism."
    )

    image = resize_if_needed(base64_to_pil(job_input["image"]))

    generator = (
        torch.Generator(device="cuda").manual_seed(42)
        if DEVICE == "cuda"
        else None
    )

    with torch.inference_mode():
        result = pipe(
            image=image,
            prompt=prompt,
            width=image.width,
            height=image.height,
            guidance_scale=2.5,
            num_inference_steps=28,
            generator=generator,
        ).images[0]

    return {
        "image": pil_to_base64(result)
    }

# ------------------------------------------------------------------
# START SERVERLESS
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting FLUX Kontext Serverless Worker")
    runpod.serverless.start({"handler": handler})
