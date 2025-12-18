import base64
import io
import os
import runpod
import torch
import random
from PIL import Image
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
DTYPE = torch.float16
MAX_SIDE = 1024
MAX_SEED = 2**31 - 1

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
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(base64_str)))
    img.load()
    return img.convert("RGB")

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
        "[photo content], remove any watermark text or logos from the image while "
        "preserving the background, texture, lighting, and overall realism. "
        "Ensure the edited areas blend seamlessly with surrounding details, "
        "leaving no visible traces of watermark removal."
    )

    guidance_scale = float(job_input.get("guidance_scale", 2.5))
    steps = int(job_input.get("steps", 28))

    seed = job_input.get("seed", 42)
    randomize_seed = bool(job_input.get("randomize_seed", False))

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    image = resize_if_needed(base64_to_pil(job_input["image"]))

    generator = (
        torch.Generator(device="cuda").manual_seed(seed)
        if DEVICE == "cuda"
        else None
    )

    with torch.inference_mode():
        result = pipe(
            image=image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            width=image.width,
            height=image.height,
            generator=generator,
        ).images[0]

    return {
        "image": pil_to_base64(result),
        "seed": seed,
    }

# ------------------------------------------------------------------
# START SERVERLESS
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting FLUX Kontext Serverless Worker")
    runpod.serverless.start({"handler": handler})
