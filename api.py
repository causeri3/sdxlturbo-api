from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
import io
import numpy as np

from PIL import Image
from sdxlturbo import generate_image_list, PROMPT

app = FastAPI()

@app.post("/sdxlturbo")
async def generate(file: UploadFile, prompt: str = Form(PROMPT)):
    pil_image = Image.open(file.file).convert("RGB")
    images = generate_image_list(pil_image, prompt=prompt)
    numpy_images = np.stack([np.array(img) for img in images])

    buffer = io.BytesIO()
    np.save(buffer, numpy_images)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="application/octet-stream")