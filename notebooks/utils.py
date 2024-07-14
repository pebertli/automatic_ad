import numpy as np
from PIL import Image

class CanvasTemplate:
    def __init__(self, canvas_size, elements):
        self.canvas_size = canvas_size
        self.elements = elements       

def maximal_rectangle(canvas_size, elements):
    top_left_canvas, bottom_right_canvas = canvas_size
    rows = bottom_right_canvas[1] - top_left_canvas[1] + 1
    cols = bottom_right_canvas[0] - top_left_canvas[0] + 1
    
    # Initialize the canvas with all zeroes
    matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Mark the obstructed areas with ones
    for element in elements:
        top_left, bottom_right = element
        for i in range(top_left[1] - top_left_canvas[1], bottom_right[1] - top_left_canvas[1] + 1):
            for j in range(top_left[0] - top_left_canvas[0], bottom_right[0] - top_left_canvas[0] + 1):
                if 0 <= i < rows and 0 <= j < cols:
                    matrix[i][j] = 1
    
    return find_maximal_rectangle(matrix)

def find_maximal_rectangle(matrix):
    if not matrix:
        return (0, 0), (0, 0)

    max_area = 0
    max_rect = (0, 0, 0, 0)
    rows = len(matrix)
    cols = len(matrix[0])
    height = [0] * cols

    for i in range(rows):
        for j in range(cols):
            height[j] = height[j] + 1 if matrix[i][j] == 0 else 0
        
        area, rect = largest_rectangle_area(height)
        if area > max_area:
            max_area = area
            max_rect = (rect[0], i - rect[1] + 1, rect[2], i)

    top_left = (max_rect[0], max_rect[1])
    bottom_right = (max_rect[2], max_rect[3])
    return top_left, bottom_right

def largest_rectangle_area(heights):
    stack = []
    max_area = 0
    rect = (0, 0, 0, 0)
    heights.append(0)
    
    for i in range(len(heights)):
        while stack and heights[i] < heights[stack[-1]]:
            h = heights[stack.pop()]
            w = i if not stack else i - stack[-1] - 1
            area = h * w
            if area > max_area:
                max_area = area
                rect = (stack[-1] + 1 if stack else 0, h, i - 1)
        stack.append(i)
    
    heights.pop()
    return max_area, rect


def new_center(image_path, roi, window, canvas_size):
    image = Image.open(image_path)
    img_width, img_height = image.size

    roi_left, roi_top, roi_right, roi_bottom = roi
    win_left, win_top, win_right, win_bottom = window
    canvas_width, canvas_height = canvas_size

    roi_center_x = (roi_left + roi_right) / 2
    roi_center_y = (roi_top + roi_bottom) / 2    
    win_center_x = (win_left + win_right) / 2
    win_center_y = (win_top + win_bottom) / 2
    
    offset_x = int(win_center_x - roi_center_x)
    offset_y = int(win_center_y - roi_center_y)

    paste_x = int(canvas_width / 2 - img_width / 2 + offset_x)
    paste_y = int(canvas_height / 2 - img_height / 2 + offset_y)

    return (paste_x, paste_y)

def resize_and_reposition(image_path, roi, window, canvas_size):
    image = Image.open(image_path)
    img_width, img_height = image.size

    roi_left, roi_top, roi_right, roi_bottom = roi
    win_left, win_top, win_right, win_bottom = window
    canvas_width, canvas_height = canvas_size
    final_canvas = Image.new('RGB', (canvas_width, canvas_height), "white")

    roi_center_x = (roi_left + roi_right) / 2
    roi_center_y = (roi_top + roi_bottom) / 2    
    win_center_x = (win_left + win_right) / 2
    win_center_y = (win_top + win_bottom) / 2
    
    offset_x = int(win_center_x - roi_center_x)
    offset_y = int(win_center_y - roi_center_y)

    scale_x = canvas_width / (roi_right - roi_left)
    scale_y = canvas_height / (roi_bottom - roi_top)
    scale = max(scale_x, scale_y)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    paste_x = int(win_center_x - (roi_center_x * scale))
    paste_y = int(win_center_y - (roi_center_y * scale))
    paste_x = min(0, max(paste_x, canvas_width - new_width))
    paste_y = min(0, max(paste_y, canvas_height - new_height))

    final_canvas.paste(resized_image, (paste_x, paste_y))

    return final_canvas