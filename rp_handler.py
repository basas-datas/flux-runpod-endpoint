import base64
import io
import runpod
import torch
import os
from PIL import Image
from diffusers import FluxKontextPipeline

# ----------------------------
# ИНИЦИАЛИЗАЦИЯ (выполняется 1 раз)
# ----------------------------

HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
).to(DEVICE)

pipe.load_lora_weights(
    "prithivMLmods/Kontext-Watermark-Remover",
    weight_name="Kontext-Watermark-Remover.safetensors",
    adapter_name="watermark_remover"
)

pipe.set_adapters(["watermark_remover"], adapter_weights=[1.0])

# ----------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ----------------------------

def base64_to_pil(base64_str: str) -> Image.Image:
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def pil_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ----------------------------
# HANDLER
# ----------------------------

def handler(job):
    """
    job["input"] ожидает:
    {
        "image": "<base64 PNG/JPEG>",
        "prompt": "удали субтитры"
    }
    """

    job_input = job["input"]

    if "image" not in job_input:
        return {"error": "image is required (base64)"}

    prompt = job_input.get(
        "prompt",
        "remove any subtitles, watermarks, timestamps, or text overlays from the image"
    )

    image = base64_to_pil(job_input["image"])

    result = pipe(
        image=image,
        prompt=prompt,
        width=image.width,
        height=image.height,
        guidance_scale=2.5,
        num_inference_steps=28,
        generator=torch.Generator(device=DEVICE).manual_seed(42),
    ).images[0]

    return {
        "image": pil_to_base64(result)
    }

# ----------------------------
# ЗАПУСК SERVERLESS
# ----------------------------

runpod.serverless.start({
    "handler": handler
})
