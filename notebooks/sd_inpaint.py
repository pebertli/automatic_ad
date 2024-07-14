import PIL
import torch

from diffusers import AutoPipelineForInpainting

# a function to extend the image and mask
def extend_image(image, origin, canvas_size):
    # add padding to the image
    canvas = PIL.Image.new("RGB", canvas_size, (255, 255, 255))    
    canvas.paste(image, origin)
    image_extended = canvas.copy()
    black_canvas = PIL.Image.new("RGB", image.size, (0, 0, 0))    
    canvas.paste(black_canvas, origin)
    mask = canvas

    return image_extended, mask

# a class for inpainting using the stable diffusion model  
class Inpainting:
    def __init__(self):
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda")    
    
    def inpaint(self, prompt, image, mask, width, height):
        image = self.pipe(prompt=prompt, image=image, mask_image=mask, strength = 1.0, width=width, height = height, num_inference_steps=25).images[0]
        return image