# Используем стабильный образ от RunPod или NVIDIA
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Установка зависимостей
RUN pip3 install --upgrade pip
RUN pip3 install diffusers==0.30.2 \
    transformers==4.44.0 \
    accelerate==0.33.0 \
    sentencepiece \
    runpod \
    huggingface_hub \
    peft

# Предзагрузка модели, чтобы не качать её при каждом запуске
RUN python3 -c "from diffusers import FluxKontextPipeline; import torch; \
    FluxKontextPipeline.from_pretrained('black-forest-labs/FLUX.1-Kontext-dev', torch_dtype=torch.bfloat16)"

WORKDIR /app
COPY handler.py .

CMD ["python3", "-u", "handler.py"]