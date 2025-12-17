import runpod
import torch
from diffusers import FluxKontextPipeline
from PIL import Image
import base64
import io

# Инициализируем модель глобально
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", 
    torch_dtype=torch.bfloat16
).to("cuda")

# Загружаем LoRA
pipe.load_lora_weights(
    "prithivMLmods/Kontext-Watermark-Remover",
    weight_name="Kontext-Watermark-Remover.safetensors"
)

def handler(job):
    try:
        job_input = job["input"]
        
        # Конвертация входящего base64 в Image
        image_data = base64.b64decode(job_input["image_base64"])
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Генерация
        output = pipe(
            prompt=job_input.get("prompt", "remove watermark"),
            image=input_image,
            num_inference_steps=28,
            guidance_scale=3.5
        ).images[0]

        # Конвертация результата в base64
        buffered = io.BytesIO()
        output.save(buffered, format="PNG")
        
        return {"image_base64": base64.b64encode(buffered.getvalue()).decode()}
    
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})