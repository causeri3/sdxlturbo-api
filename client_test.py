# uvicorn api:app --host 0.0.0.0 --port 8000 --reload

import requests
import numpy as np
import io
from PIL import Image
import time

your_file_path = r"Downloads/img1.jpg"

files = {'file': open(your_file_path, 'rb')}
data = {
    'prompt': 'dmt',
    'amount_pics': 4,
    'num_inference_steps': 15,
    'strength_min': 0.1,
    'strength_max': 0.5,
    'guidance_scale': 4
}

start = time.time()
response = requests.post(
    "http://localhost:8000/sdxlturbo",
                         files=files,
                         data=data,
                         timeout=600)

buffer = io.BytesIO(response.content)
images = np.load(buffer)
print(f"took {time.time() - start} sec ")


for idx, img_array in enumerate(images):
    img = Image.fromarray(img_array.astype(np.uint8))
    img.save(f"output_{idx}.jpg")
