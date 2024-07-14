import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.ndimage import filters
import torch.nn.functional as F
import torch
from torch import nn
import clip
import cv2


def get_region(image_path, image_caption, clip_arch, device, saliency_layer, blur=True):
    clip_model, clip_preprocess = clip.load(clip_arch, device=device, jit=False)    
    original_img = Image.open(image_path).convert('RGB')
    image_input = clip_preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    text_input = clip.tokenize([image_caption]).to(device)
    image_np = load_image(image_path, clip_model.visual.input_resolution)

    attn_map = gradCAM(
        clip_model.visual,
        image_input,
        clip_model.encode_text(text_input).float(),
        getattr(clip_model.visual, saliency_layer)
    )

    attn_map = attn_map.squeeze().detach().cpu().numpy()

    attn_img = getAttMap(image_np, attn_map, blur)
    attn_map = F.interpolate(
            torch.tensor(attn_map).unsqueeze(0).unsqueeze(0),
            (original_img.size[1], original_img.size[0]),
            mode='bicubic',
            align_corners=False).squeeze().numpy()
    bb = get_bounding_box_square(attn_map, threshold=0.05)

    attn_img = Image.fromarray((attn_img * 255).astype(np.uint8))
    attn_img  = attn_img.resize(original_img.size, Image.BICUBIC)
    if bb is not None:
        x_min, y_min, x_max, y_max = bb
        draw = ImageDraw.Draw(attn_img)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="white", width=3)
    
    return attn_img, bb

def normalize(x: np.ndarray) -> np.ndarray:
    # Normalize to [0, 1].
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x

# Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
def getAttMap(img, attn_map, blur=True):
    if blur:
        attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
    attn_map = normalize(attn_map)
    cmap = plt.get_cmap('jet')
    attn_map_c = np.delete(cmap(attn_map), 3, 2)
    attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
            (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c        

    return attn_map

def highlight_most_activated_region(attn_map):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(attn_map, cv2.COLOR_BGR2HSV)

    # Define range for heatmap colors (assuming it to be a certain range of colors)
    # This range might need adjustment based on the specific heatmap colors
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([50, 255, 255])

    # Create a mask for the heatmap region
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Apply the mask to get the heatmap area
    heatmap_area = cv2.bitwise_and(attn_map, attn_map, mask=mask)

    # Convert the heatmap area to grayscale
    gray_heatmap = cv2.cvtColor(heatmap_area, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_heatmap = cv2.GaussianBlur(gray_heatmap, (5, 5), 0)

    # Apply Otsu's thresholding to the heatmap region
    _, thresholded_heatmap = cv2.threshold(blurred_heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the thresholded heatmap region
    heatmap_contours, _ = cv2.findContours(thresholded_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the maximum area in the heatmap region
    max_heatmap_contour = max(heatmap_contours, key=cv2.contourArea)

    # Create a mask for the most activated region in the heatmap
    heatmap_mask = np.zeros_like(gray_heatmap)

    return heatmap_mask

def get_bounding_box_square(attn_map, threshold=0.5):
    # indices = np.where(attn_map >= threshold)
    # blur the image to get a smoother heatmap
    attn_map = filters.gaussian_filter(attn_map, 0.07*max(attn_map.shape[:2]))    
    # instead of using the threshold, we will use the top 5% of the pixels with the highest values
    indices = np.where(attn_map >= np.percentile(attn_map, 95))    
    
    
    if len(indices[0]) == 0 or len(indices[1]) == 0:
        return None  # No region found

    x_min, x_max = indices[1].min(), indices[1].max()
    y_min, y_max = indices[0].min(), indices[0].max()
    
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
    if square_x_max >= attn_map.shape[1]:
        offset = square_x_max - attn_map.shape[1] + 1
        square_x_min -= offset
        square_x_max -= offset
    if square_y_min < 0:
        offset = -square_y_min
        square_y_min += offset
        square_y_max += offset
    if square_y_max >= attn_map.shape[0]:
        offset = square_y_max - attn_map.shape[0] + 1
        square_y_min -= offset
        square_y_max -= offset

    # Ensure the bounding box is square and within bounds
    square_x_min = max(0, square_x_min)
    square_x_max = min(attn_map.shape[1] - 1, square_x_max)
    square_y_min = max(0, square_y_min)
    square_y_max = min(attn_map.shape[0] - 1, square_y_max)

    # Adjust the size if the bounding box is out of bounds
    if square_x_max - square_x_min != square_y_max - square_y_min:
        size = min(square_x_max - square_x_min, square_y_max - square_y_min)
        square_x_max = square_x_min + size
        square_y_max = square_y_min + size

    return (square_x_min, square_y_min, square_x_max, square_y_max)

    
def load_image(img_path, resize=None):
    image = Image.open(img_path).convert("RGB")
    if resize is not None:
        image = image.resize((resize, resize))
    return np.asarray(image).astype(np.float32) / 255.


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()
        
    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
        
    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:        
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()
    
        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)
    
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
        
    return gradcam