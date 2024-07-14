from transformers import AutoProcessor, AutoModelForCausalLM  
from PIL import Image, ImageDraw
import os
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import re

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

def florence_caption(image_path):
    model_id = 'microsoft/Florence-2-large'
    # not using cuda since it requires flash_attn and we might have issues installing it
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
            model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa" ,trust_remote_code=True).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=512,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    generated_text = processor.post_process_generation(
        generated_text, 
        task="<DETAILED_CAPTION>", 
        image_size=(image.width, image.height)
    )
    generated_text = generated_text["<DETAILED_CAPTION>"]
    # usin regex to remove any <loc_*>
    generated_text = re.sub(r'<loc_.*?>', '', generated_text)


    return generated_text

def florence_inference_region(image_path, prompt):
    model_id = 'microsoft/Florence-2-large'
    # not using cuda since it requires flash_attn and we might have issues installing it
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports): #workaround for unnecessary flash_attn requirement
            model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="sdpa" ,trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=512,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task="<CAPTION_TO_PHRASE_GROUNDING>", 
        image_size=(image.width, image.height)
    )

    # return is like this {'<CAPTION_TO_PHRASE_GROUNDING>': {'bboxes': [[34.23999786376953, 159.1199951171875, 582.0800170898438, 374.6399841308594], [1.5999999046325684, 4.079999923706055, 639.0399780273438, 305.03997802734375]], 'labels': ['A green car', 'a yellow building']}}
    # get only the bboxes
    bbs = parsed_answer['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
    # calculate the bounding box of all the bbs
    x_min = min([bb[0] for bb in bbs])
    y_min = min([bb[1] for bb in bbs])
    x_max = max([bb[2] for bb in bbs])
    y_max = max([bb[3] for bb in bbs])    

    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    bb_size = max(x_max - x_min, y_max - y_min)

    # Calculate new square bounding box coordinates
    half_size = (bb_size + 1) // 2
    square_x_min = x_center - half_size
    square_x_max = x_center + half_size
    square_y_min = y_center - half_size
    square_y_max = y_center + half_size

    # Adjust the square bounding box to ensure it is within the image boundaries
    if square_x_min < 0:
        offset = -square_x_min
        square_x_min += offset
        square_x_max += offset
    if square_x_max >= image.size[1]:
        offset = square_x_max - image.size[1] + 1
        square_x_min -= offset
        square_x_max -= offset
    if square_y_min < 0:
        offset = -square_y_min
        square_y_min += offset
        square_y_max += offset
    if square_y_max >= image.size[0]:
        offset = square_y_max - image.size[0] + 1
        square_y_min -= offset
        square_y_max -= offset

    # Ensure the bounding box is square and within bounds
    square_x_min = max(0, square_x_min)
    square_x_max = min(image.size[1] - 1, square_x_max)
    square_y_min = max(0, square_y_min)
    square_y_max = min(image.size[0] - 1, square_y_max)

    # Adjust the size if the bounding box is out of bounds
    if square_x_max - square_x_min != square_y_max - square_y_min:
        size = min(square_x_max - square_x_min, square_y_max - square_y_min)
        square_x_max = square_x_min + size
        square_y_max = square_y_min + size

    # using imagedraw to draw the bounding box
    draw = ImageDraw.Draw(image)
    draw.rectangle([square_x_min, square_y_min, square_x_max, square_y_max], outline="white", width=3)
       

    return image, (square_x_min, square_y_min, square_x_max, square_y_max)    
