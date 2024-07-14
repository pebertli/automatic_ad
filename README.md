# Session 3

This repository contains three Jupyter Notebooks the Task Coding. Each notebook applies a unique method to handle specific image processing challenges.
Also, the Powerpoint presentation and the pipeline proposed for the Task Design are included in the repository

## Notebooks

### 1. task_coding_gradcam.ipynb
This notebook uses the Grad-CAM (Gradient-weighted Class Activation Mapping) technique to address the problem as requested. Grad-CAM is used to visualize the region of interest in the input image by highlighting the important regions that contribute to the model's decision-making process.

#### Key Features:
- Implements Grad-CAM to identify and visualize regions of interest.
- Provides detailed explanations and visualizations of the Grad-CAM process.
- Includes generated outputs for reference.

### 2. task_coding_florence.ipynb
This notebook uses the Florence-2 method to detect the region of interest in the input image. Florence-2 is an alternative technique to Grad-CAM and provides a different perspective on identifying important regions in images.

#### Key Features:
- Utilizes the Florence-2 method for region of interest detection.
- Contains detailed explanations and visualizations.
- Includes generated outputs for reference.

### 3. task_coding_outpaint.ipynb
This notebook addresses the task of outpainting, where it fills in the missing parts of the input image to create a complete and meaningful canvas. Unlike other methods that resize images to fit the window, outpainting extends the image contextually.

#### Key Features:
- Implements outpainting to fill missing parts of images.
- Ensures the whole canvas is filled meaningfully without resizing.
- Provides detailed explanations and visualizations of the outpainting process.
- Includes generated outputs for reference.

## How to Use
1. Clone this repository to your local machine.   
   git clone https://github.com/your-repository/image-processing-project.git
2. Open the desired notebook in Jupyter Notebook or Google Colab.
3. Run the notebook cells to execute the code and view the results.

Models should be downloaded from public repositories automatically.