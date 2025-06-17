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
#pipe.to("cuda")
pipe.to("mps")


# ___________ PARAMS _________________ #
# Official Documenation
# When using SDXL-Turbo for image-to-image generation, make sure that num_inference_steps * strength is larger or equal to 1.
# The image-to-image pipeline will run for int(num_inference_steps * strength) steps, e.g. 0.5 * 2.0 = 1 step in our example below.
NUM_INFERENCE_STEPS=25
STRENGTH_MIN = 0.05
STRENGTH_MAX = 0.6
GUIDANCE_SCALE=8
PROMPT = "DMT"


REZ = (512, 512)
AMOUNT_PICS = 22
# ___________ YALLA _________________ #

def generate_image_list(pil_image, prompt:str = PROMPT):
    init_image = load_image(pil_image).resize(REZ)
    start_time = time()
    images_list = []
    images_list.append(init_image)
    image = pipe(prompt,
             image=init_image,
             #num_inference_steps=math.ceil(1/STRENGTH_MIN),
            num_inference_steps = NUM_INFERENCE_STEPS,
             strength=STRENGTH_MIN,
             guidance_scale=GUIDANCE_SCALE).images[0]
    logging.info(f"No 1")
    logging.info("It took {:.2f} Sec".format((time() - start_time)))

    images_list.append(image)

    for i in range(AMOUNT_PICS):
        # linear increasing
        strength = (STRENGTH_MIN + (STRENGTH_MAX - STRENGTH_MIN) / AMOUNT_PICS * i)

        # geometric increasing
        # strength = STRENGTH_MIN * ((STRENGTH_MAX / STRENGTH_MIN) ** (i / (AMOUNT_PICS - 1)))
        logging.debug(f"Strength: {strength}")

        start_time_one_pic = time()
        image = pipe(prompt, image=image,
                     #num_inference_steps=math.ceil(1/strength),
                     num_inference_steps=NUM_INFERENCE_STEPS,
                     strength=strength,
                     guidance_scale=GUIDANCE_SCALE).images[0]
        image = ImageOps.autocontrast(image, cutoff=5)
        logging.info(f"No {i + 2}")
        logging.info("It took {:.2f} Sec".format((time() - start_time_one_pic)))
        images_list.append(image)

    logging.info("It took {:.2f} Sec for {} images".format((time() - start_time), AMOUNT_PICS))
    return images_list