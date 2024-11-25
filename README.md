# Enhanced Image Processing Application

This application allows users to upload an image and apply various image processing techniques interactively using **Streamlit** and **OpenCV**. The app includes various options for processing the image, including edge detection, filtering, contrast enhancement, and more. Additionally, users can compare the results of each technique with the original image.

## Features

- **Image Upload**: Allows users to upload an image file (JPG, JPEG, PNG).
- **Processing Options**: A range of image processing techniques can be applied to the uploaded image:
  - **Histogram Equalization**: Enhances the contrast of the image.
  - **Error Diffusion Halftoning**: Converts the image to a halftone using error diffusion.
  - **Edge Detection**: Multiple edge detection techniques are available, including Sobel, Prewitt, and Kirsch operators.
  - **Homogeneity Operator**: Detects areas with homogeneous pixel values.
  - **Difference Operator**: Highlights edges based on pixel differences.
  - **Difference of Gaussians (DoG)**: Edge detection based on the difference of two Gaussian-blurred images.
  - **Contrast-Based Edge Detection**: Detects edges based on contrast changes.
  - **Variance Operator**: Detects edges based on pixel variance.
  - **Range Operator**: Highlights edges based on the range of pixel values.
  - **Convolution**: Apply high-pass or low-pass filters using convolution.
  - **Median Filter**: Removes noise using a median filter.
  - **Image Addition/Subtraction**: Allows addition or subtraction of two images.
  - **Invert Image**: Inverts the pixel values of the image.
  - **Cut and Paste**: Cut a portion of one image and paste it onto another image.

## How to Use

1. **Upload an Image**: Click the "Choose an image file" button and upload an image in JPG, JPEG, or PNG format.
2. **Select Processing Technique**: In the sidebar, select the desired processing technique from the dropdown list.
3. **Adjust Parameters (if applicable)**: Some processing techniques allow you to adjust parameters such as thresholds or filter types. Use the sliders provided to fine-tune the effect.
4. **View Results**: After processing, the app will display the processed image and a histogram of the pixel values. If applicable, the app will also show comparisons between the original and processed image.

## Processing Techniques

- **Histogram Equalization**: Enhances the contrast of the image by redistributing pixel intensity values.
- **Edge Detection (Sobel, Prewitt, Kirsch)**: Detects edges in the image using different edge detection filters. You can view edges in different directions (e.g., horizontal, vertical).
- **Error Diffusion Halftoning**: Reduces the number of colors in an image by spreading error information between adjacent pixels.
- **Difference of Gaussians (DoG)**: Applies Gaussian blurring twice with different sigma values and subtracts the results to highlight edges.
- **Median Filter**: Removes noise from the image by replacing each pixel's value with the median of the surrounding pixels.
- **Image Addition/Subtraction**: Allows you to add or subtract pixel values between two grayscale images.
- **Invert Image**: Inverts the pixel intensity values of the image (i.e., transforms black to white and vice versa).
- **Cut and Paste**: Cut a selected region from one image and paste it onto another image.

## Installation

To run this application locally, you will need to install the required dependencies. Follow the steps below:

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/image-processing-streamlit.git
cd image-processing-streamlit
```

### Step 2: Create a Virtual Environment (Optional but Recommended)

- It is recommended to create a virtual environment to manage your dependencies separately from the system installation.

For macOS/Linux:
```bash
python3 -m venv env
source env/bin/activate
```
For Windows:
```bash
python -m venv env
env\Scripts\activate
```
### Step 3: Install the required dependencies

-With the virtual environment activated, install the dependencies listed in the requirements.txt file:
```bash
pip install -r requirements.txt
```

### Step 4: Run the application
-Once the dependencies are installed, you can run the application using Streamlit:
```bash
streamlit run app.py
```
-This will start the application and open it in your default web browser. You should now be able to interact with the image processing features.

### Step 5: Open the application in the browser
-Once the app is running, open your web browser and navigate to:
```bash
http://localhost:8501
```
Here, you can upload an image and start using the various image processing techniques available.


