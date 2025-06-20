from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch
from PIL import ImageOps
from time import time
import logging

# ___________ LOAD MODEL _________________ #
pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo",
                                                  torch_dtype=torch.float16,
                                                  variant="fp16"
                                                  )
pipe.to("cuda")
#pipe.to("mps")


# ___________ PARAMS _________________ #
# Official Documenation
# When using SDXL-Turbo for image-to-image generation, make sure that num_inference_steps * strength is larger or equal to 1.
# The image-to-image pipeline will run for int(num_inference_steps * strength) steps, e.g. 0.5 * 2.0 = 1 step in our example below.
REZ = (512, 512)

NUM_INFERENCE_STEPS=50
STRENGTH_MIN = 0.05
STRENGTH_MAX = 0.35
GUIDANCE_SCALE=8
PROMPT = "DMT"
AMOUNT_PICS = 16
# ___________ YALLA _________________ #

def generate_image_list(pil_image, 
                        prompt:str = PROMPT,
                        amount_pics:int = AMOUNT_PICS,
                        num_inference_steps:int = NUM_INFERENCE_STEPS,
                        strength_min:float = STRENGTH_MIN,
                        strength_max:float = STRENGTH_MAX,
                        guidance_scale:int = GUIDANCE_SCALE):
    init_image = load_image(pil_image).resize(REZ)
    start_time = time()
    images_list = []
    images_list.append(init_image)
    image = pipe(prompt,
             image=init_image,
             #num_inference_steps=math.ceil(1/strength_min),
            num_inference_steps = num_inference_steps,
             strength=strength_min,
             guidance_scale=guidance_scale).images[0]
    logging.info(f"No 1")
    logging.info("It took {:.2f} Sec".format((time() - start_time)))

    images_list.append(image)

    for i in range(amount_pics):
        # linear increasing
        strength = (strength_min + (strength_max - strength_min) / amount_pics * i)

        # geometric increasing
        # strength = strength_min * ((strength_max / strength_min) ** (i / (amount_pics - 1)))
        logging.debug(f"Strength: {strength}")

        start_time_one_pic = time()
        image = pipe(prompt, image=image,
                     #num_inference_steps=math.ceil(1/strength),
                     num_inference_steps=num_inference_steps,
                     strength=strength,
                     guidance_scale=guidance_scale).images[0]
        image = ImageOps.autocontrast(image, cutoff=5)
        logging.info(f"No {i + 2}")
        logging.info("It took {:.2f} Sec".format((time() - start_time_one_pic)))
        images_list.append(image)

    logging.info("It took {:.2f} Sec for {} images".format((time() - start_time), amount_pics))
    return images_list