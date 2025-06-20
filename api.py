from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
import io
import numpy as np

from PIL import Image
from sdxlturbo import (generate_image_list,
                       NUM_INFERENCE_STEPS,
                       STRENGTH_MIN,
                       STRENGTH_MAX,
                       GUIDANCE_SCALE,
                       PROMPT,
                       AMOUNT_PICS)

app = FastAPI()
@app.post("/sdxlturbo")
async def generate(
    file: UploadFile,
    prompt: str = Form(PROMPT),
    amount_pics: int = Form(AMOUNT_PICS),
    num_inference_steps: int = Form(NUM_INFERENCE_STEPS),
    strength_min: float = Form(STRENGTH_MIN),
    strength_max: float = Form(STRENGTH_MAX),
    guidance_scale: int = Form(GUIDANCE_SCALE)
):
    pil_image = Image.open(file.file).convert("RGB")
    images = generate_image_list(
        pil_image,
        prompt=prompt,
        amount_pics=amount_pics,
        num_inference_steps=num_inference_steps,
        strength_min=strength_min,
        strength_max=strength_max,
        guidance_scale=guidance_scale
    )
    numpy_images = np.stack([np.array(img) for img in images])

    buffer = io.BytesIO()
    np.save(buffer, numpy_images)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="application/octet-stream")