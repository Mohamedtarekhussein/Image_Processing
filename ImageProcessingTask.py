import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def create_histogram(image):
    """Calculate histogram from scratch with additional statistics"""
    if len(image.shape) == 3:
        hist_r = np.zeros(256)
        hist_g = np.zeros(256)
        hist_b = np.zeros(256)
        mean_r = mean_g = mean_b = 0
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                hist_r[image[i,j,0]] += 1
                hist_g[image[i,j,1]] += 1
                hist_b[image[i,j,2]] += 1
                mean_r += image[i,j,0]
                mean_g += image[i,j,1]
                mean_b += image[i,j,2]
        
        total_pixels = image.shape[0] * image.shape[1]
        stats = {
            'mean': (mean_r/total_pixels, mean_g/total_pixels, mean_b/total_pixels),
            'max': (hist_r.max(), hist_g.max(), hist_b.max()),
            'min': (hist_r.min(), hist_g.min(), hist_b.min())
        }
        return (hist_r, hist_g, hist_b), stats
    else:
        hist = np.zeros(256)
        mean = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                hist[image[i,j]] += 1
                mean += image[i,j]
        
        total_pixels = image.shape[0] * image.shape[1]
        stats = {
            'mean': mean/total_pixels,
            'max': hist.max(),
            'min': hist.min()
        }
        return hist, stats

def histogram_equalization(image):
    """Enhanced histogram equalization with contrast metrics"""
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    
    # Calculate histogram and CDF
    hist = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[image[i,j]] += 1
    
    # Calculate cumulative distribution function (CDF)
    cdf = np.zeros(256)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]
    
    # Normalize CDF
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    
    # Create equalized image with contrast metrics
    equalized = np.zeros_like(image)
    contrast_before = np.std(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            equalized[i,j] = cdf[image[i,j]]
    
    contrast_after = np.std(equalized)
    metrics = {
        'contrast_before': contrast_before,
        'contrast_after': contrast_after,
        'improvement': (contrast_after - contrast_before) / contrast_before * 100
    }
    
    return equalized.astype(np.uint8), metrics

def error_diffusion(image, threshold=128):
    """Simplified error diffusion with Floyd-Steinberg pattern"""
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    
    output = np.zeros_like(image, dtype=np.uint8)
    error = np.zeros_like(image, dtype=float)
    padded = np.pad(image.astype(float), ((0,1), (0,1)), 'constant')
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            old_pixel = padded[i,j] + error[i,j]
            new_pixel = 255 if old_pixel > threshold else 0
            output[i,j] = new_pixel
            quant_error = old_pixel - new_pixel
            
            # Distribute error using Floyd-Steinberg pattern
            if j + 1 < image.shape[1]:
                error[i, j+1] += quant_error * 7/16
            if i + 1 < image.shape[0]:
                if j > 0:
                    error[i+1, j-1] += quant_error * 3/16
                error[i+1, j] += quant_error * 5/16
                if j + 1 < image.shape[1]:
                    error[i+1, j+1] += quant_error * 1/16
    
    return output

def sobel_edge_detection(image):
    """Implement Sobel edge detection"""
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    edges_x = np.zeros_like(image)
    edges_y = np.zeros_like(image)
    
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            neighborhood = image[i-1:i+2, j-1:j+2]
            edges_x[i,j] = np.abs(np.sum(neighborhood * sobel_x))
            edges_y[i,j] = np.abs(np.sum(neighborhood * sobel_y))
    
    return {
        'horizontal': edges_x.astype(np.uint8),
        'vertical': edges_y.astype(np.uint8)
    }

def plot_comparison(original, processed, title):
    """Enhanced comparison plot with histograms and statistics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    orig_hist, orig_stats = create_histogram(original)
    proc_hist, proc_stats = create_histogram(processed)
    
    if isinstance(orig_hist, tuple):
        ax3.plot(orig_hist[0], 'r', alpha=0.5, label='Red')
        ax3.plot(orig_hist[1], 'g', alpha=0.5, label='Green')
        ax3.plot(orig_hist[2], 'b', alpha=0.5, label='Blue')
        ax3.legend()
    else:
        ax3.plot(orig_hist, 'gray', label='Intensity')
        ax3.legend()
    
    ax3.set_title('Original Histogram')
    ax3.set_xlabel('Pixel Value')
    ax3.set_ylabel('Frequency')
    
    ax2.imshow(processed, cmap='gray')
    ax2.set_title(f'Processed Image ({title})')
    ax2.axis('off')
    
    if isinstance(proc_hist, tuple):
        ax4.plot(proc_hist[0], 'r', alpha=0.5, label='Red')
        ax4.plot(proc_hist[1], 'g', alpha=0.5, label='Green')
        ax4.plot(proc_hist[2], 'b', alpha=0.5, label='Blue')
        ax4.legend()
    else:
        ax4.plot(proc_hist, 'gray', label='Intensity')
        ax4.legend()
    
    ax4.set_title('Processed Histogram')
    ax4.set_xlabel('Pixel Value')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig, orig_stats, proc_stats

def plot_edge_detection(original, results):
    """Plot edge detection results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax3.imshow(results['horizontal'], cmap='gray')
    ax3.set_title('Horizontal Edges')
    ax3.axis('off')
    
    ax4.imshow(results['vertical'], cmap='gray')
    ax4.set_title('Vertical Edges')
    ax4.axis('off')
    
    plt.tight_layout()
    return fig

def prewitt_edge_detection(image):
    """Prewitt edge detection with all directions"""
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    
    # Prewitt kernels for all directions
    kernels = {
        'horizontal': np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]]),
        'vertical': np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]]),
        'diagonal_45': np.array([[0, 1, 1],
                            [-1, 0, 1],
                            [-1, -1, 0]]),
        'diagonal_135': np.array([[1, 1, 0],
                                [1, 0, -1],
                                [0, -1, -1]])
    }
    
    results = {}
    
    for direction, kernel in kernels.items():
        result = np.zeros_like(image, dtype=float)
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                neighborhood = image[i-1:i+2, j-1:j+2]
                result[i,j] = np.abs(np.sum(neighborhood * kernel))
        results[direction] = (result / result.max() * 255).astype(np.uint8)
    
    return results

def kirsch_edge_detection(image):
    """Kirsch edge detection with all 8 compass directions"""
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    
    # Kirsch kernels for all 8 compass directions
    kernels = {
        'north': np.array([[ 5,  5,  5],
                        [-3,  0, -3],
                        [-3, -3, -3]]),
        'northeast': np.array([[-3,  5,  5],
                            [-3,  0,  5],
                            [-3, -3, -3]]),
        'east': np.array([[-3, -3,  5],
                        [-3,  0,  5],
                        [-3, -3,  5]]),
        'southeast': np.array([[-3, -3, -3],
                            [-3,  0,  5],
                            [-3,  5,  5]]),
        'south': np.array([[-3, -3, -3],
                        [-3,  0, -3],
                        [ 5,  5,  5]]),
        'southwest': np.array([[-3, -3, -3],
                            [ 5,  0, -3],
                            [ 5,  5, -3]]),
        'west': np.array([[ 5, -3, -3],
                        [ 5,  0, -3],
                        [ 5, -3, -3]]),
        'northwest': np.array([[ 5,  5, -3],
                            [ 5,  0, -3],
                            [-3, -3, -3]])
    }
    
    results = {}
    
    for direction, kernel in kernels.items():
        result = np.zeros_like(image, dtype=float)
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                neighborhood = image[i-1:i+2, j-1:j+2]
                result[i,j] = np.abs(np.sum(neighborhood * kernel))
        results[direction] = (result / result.max() * 255).astype(np.uint8)
    
    return results

def plot_prewitt_results(original, results):
    """Plot Prewitt edge detection results"""
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 15))
    
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(results['horizontal'], cmap='gray')
    ax2.set_title('Horizontal')
    ax2.axis('off')
    
    ax3.imshow(results['vertical'], cmap='gray')
    ax3.set_title('Vertical')
    ax3.axis('off')
    
    ax4.imshow(results['diagonal_45'], cmap='gray')
    ax4.set_title('Diagonal 45°')
    ax4.axis('off')
    
    ax5.imshow(results['diagonal_135'], cmap='gray')
    ax5.set_title('Diagonal 135°')
    ax5.axis('off')
    
    plt.tight_layout()
    return fig

def plot_kirsch_results(original, results):
    """Plot Kirsch edge detection results"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Plot original image in center
    axes[1,1].imshow(original, cmap='gray')
    axes[1,1].set_title('Original')
    axes[1,1].axis('off')
    
    # Plot directions in their corresponding positions
    positions = {
        'north': (0,1),
        'northeast': (0,2),
        'east': (1,2),
        'southeast': (2,2),
        'south': (2,1),
        'southwest': (2,0),
        'west': (1,0),
        'northwest': (0,0)
    }
    
    for direction, pos in positions.items():
        axes[pos].imshow(results[direction], cmap='gray')
        axes[pos].set_title(direction.capitalize())
        axes[pos].axis('off')
    
    plt.tight_layout()
    return fig
# Advenced edge detection __>
# Homogeneity Operator function
def homogeneity_operator(image, threshold):
    # Ensure the image is grayscale (2D array)
    if len(image.shape) == 3:  # If the image is RGB (3 channels)
        image = np.mean(image, axis=2)  # Convert RGB to grayscale by averaging channels
    
    height, width = image.shape
    homogeneity_image = np.zeros_like(image)
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            center_pixel = image[i, j]
            differences = [
                abs(center_pixel - image[i-1, j-1]),
                abs(center_pixel - image[i-1, j]),
                abs(center_pixel - image[i-1, j+1]),
                abs(center_pixel - image[i, j-1]),
                abs(center_pixel - image[i+1, j-1]),
                abs(center_pixel - image[i+1, j]),
                abs(center_pixel - image[i+1, j+1]),
            ]
            homogeneity_value = max(differences)
            homogeneity_image[i, j] = homogeneity_value
            homogeneity_image[i, j] = np.where(homogeneity_image[i, j] >= threshold, homogeneity_image[i, j], 0)
    
    return homogeneity_image.astype(np.uint8) 
# difference operator 
def difference_operator(image, threshold):
    if len(image.shape) == 3:  # If the image is RGB (3 channels)
        image = np.mean(image, axis=2)  # Convert RGB to grayscale by averaging channels
    
    height, width = image.shape
    difference_image = np.zeros_like(image)
    
    # Apply the difference operator
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            diff1 = abs(image[i-1, j-1] - image[i+1, j+1])
            diff2 = abs(image[i-1, j+1] - image[i+1, j-1])
            diff3 = abs(image[i, j-1] - image[i, j+1])
            diff4 = abs(image[i-1, j] - image[i+1, j])
            max_difference = max(diff1, diff2, diff3, diff4)
            difference_image[i, j] = max_difference

            # Thresholding to create binary output
            difference_image[i, j] = np.where(difference_image[i, j] >= threshold, difference_image[i, j], 0)
    
    return difference_image.astype(np.uint8)
#difference of gaussians DOG
def difference_of_gaussians(image):
    if len(image.shape) == 3:  # If the image is RGB (3 channels)
        image = np.mean(image, axis=2)  # Convert RGB to grayscale by averaging channels
    
    mask1=np.array([
        [ 0,  0, -1, -1, -1,  0,  0],
        [ 0, -2, -3, -3, -3, -2,  0],
        [-1, -3,  5,  5,  5, -3, -1],
        [-1, -3,  5, 16,  5, -3, -1],
        [-1, -3,  5,  5,  5, -3, -1],
        [ 0, -2, -3, -3, -3, -2,  0],
        [ 0,  0, -1, -1, -1,  0,  0]
    ], dtype=np.float32)
    mask2=np.array([
        [ 0,  0,  0, -1, -1, -1,  0,  0,  0],
        [ 0, -2, -3, -3, -3, -3, -3, -2,  0],
        [ 0, -3, -2, -1, -1, -1, -2, -3,  0],
        [-1, -3, -1,  9,  9,  9, -1, -3, -1],
        [-1, -3, -1,  9, 19,  9, -1, -3, -1],
        [-1, -3, -1,  9,  9,  9, -1, -3, -1],
        [ 0, -3, -2, -1, -1, -1, -2, -3,  0],
        [ 0, -2, -3, -3, -3, -3, -3, -2,  0],
        [ 0,  0,  0, -1, -1, -1,  0,  0,  0]
    ], dtype=np.float32)
    blurred1=cv2.filter2D(image,-1,mask1)
    blurred2=cv2.filter2D(image,-1,mask2)
    
    dog=blurred1-blurred2
    dog_normalized = np.clip(dog, 0, 255).astype(np.uint8)
    
    return dog_normalized, blurred1, blurred2
#constrast_based_egde_detection
def contrast_based_edge_detection(image):
    # Convert to grayscale if the image is RGB
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    
    # Edge detection mask (Laplacian of Gaussian)
    edge_mask = np.array([[-1, 0, -1],
                          [0, -4, 0],
                          [-1, 0, -1]])
    
    # Apply edge detection (Laplacian filter)
    edge_output = cv2.filter2D(image, -1, edge_mask)
    
    # Apply Gaussian smoothing instead of the average filter
    average_output = cv2.GaussianBlur(image, (5, 5), 0)  # You can adjust the kernel size
    average_output = average_output.astype(float)
    
    # Avoid division by zero by adding a small constant
    average_output += 1e-10
    
    # Compute contrast edge detection
    contrast_edge = edge_output / average_output
    
    # Threshold to remove low-contrast areas
    contrast_edge[contrast_edge < 0.1] = 0  # Adjust threshold as needed
    
    # Ensure there are no NaN values
    contrast_edge = np.nan_to_num(contrast_edge)
    
    return contrast_edge, edge_output, average_output


###################
# variance_operator
def variance_operator(image):
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    output = np.zeros_like(image)
    height, width = image.shape[:2]  # Use only height and width for the first two dimensions

    for i in range(1, height-1):
        for j in range(1, width-1):
            neighborhood = image[i-1:i+2, j-1:j+2]
            mean = np.mean(neighborhood)
            variance = np.sum((neighborhood - mean)**2) / 9
            output[i, j] = variance
    return output
# range_operator
def range_operator(image):
    # Check if the image is colored (3 channels) or grayscale (2 channels)
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    output = np.zeros_like(image)
    height, width = image.shape  # Only unpack height and width from the first two dimensions

    for i in range(1, height-1):
        for j in range(1, width-1):
            neighborhood = image[i-1:i+2, j-1:j+2]
            range_value = np.max(neighborhood) - np.min(neighborhood)
            output[i, j] = range_value

    return output

mask_high_pass= np.array([
    [0,-1,0],[-1,5,-1],[0,-1,0]
],dtype=np.float32)
mask_low_pass= np.array([
    [0,1/6,0],[1/6,2/6,1/6],[0,1/6,0]
],dtype=np.float32)
## high pass , low pass 
def conv(image,mask1):
    result=cv2.filter2D(image,-1,mask1)
    return result

def median_filter(image):
    # Check if the image is colored (3 channels) or grayscale (2 channels)
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)  # Convert to grayscale by averaging channels

    height, width = image.shape
    filtered_image = np.zeros_like(image)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            neighborhood = image[i-1:i+2, j-1:j+2]
            median_value = np.median(neighborhood)
            filtered_image[i, j] = median_value
    return filtered_image
# image operator 
def add(image1,image2):
    height, width = image1.shape
    added_image = np.zeros((height,width),dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            added_image[i,j]=image1[i,j] + image2[i,j]
            added_image[i,j]=max(0,min(added_image[i,j],255))
    return added_image

def subtract(image1,image2):
    height, width = image1.shape
    subtracted_image = np.zeros((height,width),dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            subtracted_image[i,j]=image1[i,j] + image2[i,j]
            subtracted_image[i,j]=max(0,min(subtracted_image[i,j],255))
    return subtracted_image

def invert(image1):
    height, width = image1.shape
    inverted_image = np.zeros((height,width),dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            inverted_image[i,j]=255-image1[i,j]
    return inverted_image

def cut_paste(image1,image2,position,size):
    x,y = position
    w,h=size
    cut_image=image1[y:y+h,x:x+w]
    output_image=np.copy(image2)
    output_image[y:y+h,x:x+w]=cut_image
    
    return output_image
def main():
    st.title("Enhanced Image Processing Application")
    st.write("Upload an image to apply various processing techniques")
    
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        
        if len(image.shape) == 3:  # Check if the image has 3 channels (color image)
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image  # Already grayscale
        
        processing_option = st.sidebar.selectbox(
            "Select Processing Technique",
            ["Histogram Equalization", "Error Diffusion Halftoning", 
            "Sobel Edge Detection", "Prewitt Edge Detection", 
            "Kirsch Edge Detection","Homogeneity Operator", "Difference Operator","Difference of Gaussians (DoG)","Contrast Based Edge Detection", "Variance Operator", "Range Operator", "Convolution", "Median Filter","Add Images", "Subtract Images", "Invert Image", "Cut and Paste"]
        )
        
        if processing_option == "Histogram Equalization":
            processed, metrics = histogram_equalization(image)
            
            st.write("### Contrast Improvement Metrics")
            st.write(f"Original Contrast: {metrics['contrast_before']:.2f}")
            st.write(f"Enhanced Contrast: {metrics['contrast_after']:.2f}")
            st.write(f"Improvement: {metrics['improvement']:.2f}%")
            
            fig, orig_stats, proc_stats = plot_comparison(image, processed, "Histogram Equalization")
            st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Original Image Statistics")
                if isinstance(orig_stats['mean'], tuple):
                    st.write(f"Mean (R,G,B): {orig_stats['mean']}")
                else:
                    st.write(f"Mean: {orig_stats['mean']:.2f}")
            
            with col2:
                st.write("### Processed Image Statistics")
                if isinstance(proc_stats['mean'], tuple):
                    st.write(f"Mean (R,G,B): {proc_stats['mean']}")
                else:
                    st.write(f"Mean: {proc_stats['mean']:.2f}")
            
        elif processing_option == "Error Diffusion Halftoning":
            threshold = st.sidebar.slider("Threshold", 0, 255, 128)
            processed = error_diffusion(image, threshold)
            
            fig, orig_stats, proc_stats = plot_comparison(image, processed, "Error Diffusion")
            st.pyplot(fig)
            
        elif processing_option == "Sobel Edge Detection":
            results = sobel_edge_detection(image)
            fig = plot_edge_detection(image, results)
            st.pyplot(fig)
            
        elif processing_option == "Prewitt Edge Detection":
            results = prewitt_edge_detection(image)
            fig = plot_prewitt_results(image, results)
            st.pyplot(fig)
            
            selected_direction = st.selectbox(
                "Select direction to view",
                ['horizontal', 'vertical', 'diagonal_45', 'diagonal_135']
            )
            st.image(results[selected_direction], 
                    caption=f"{selected_direction.replace('_', ' ').title()} Edges",
                    use_column_width=True)
            
        elif processing_option == "Kirsch Edge Detection":
            results = kirsch_edge_detection(image)
            fig = plot_kirsch_results(image, results)
            st.pyplot(fig)
            
            selected_direction = st.selectbox(
                "Select direction to view",
                ['north', 'northeast', 'east', 'southeast', 
                'south', 'southwest', 'west', 'northwest']
            )
            st.image(results[selected_direction], 
                    caption=f"{selected_direction.capitalize()} Edges",
                    use_column_width=True)
            
        elif processing_option == "Homogeneity Operator":
            threshold = st.sidebar.slider("Threshold", 0, 255, 128)
            processed = homogeneity_operator(image, threshold)
            
            st.write("### Homogeneity Operator")
            st.image(processed, caption="Processed Image using Homogeneity Operator", use_column_width=True)

            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            st.pyplot(fig)
        
        elif processing_option == "Difference Operator":
            threshold = st.sidebar.slider("Threshold", 0, 255, 128)
            processed = difference_operator(image, threshold)
            
            st.write("### Difference Operator")
            st.image(processed, caption="Processed Image using Difference Operator", use_column_width=True)

            # Display histogram for the processed image
            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            st.pyplot(fig)
        elif processing_option == "Difference of Gaussians (DoG)":
                processed, blurred1, blurred2 = difference_of_gaussians(image)

                st.write("### Difference of Gaussian (DoG) Results")
                # Normalize and display image
                st.image(processed, caption="DoG Processed Image", use_column_width=True)

                st.write("### Blurred Images (Gaussian Blurring)")
                # Normalize and display image
                st.image(blurred1, caption="First Gaussian Blur", use_column_width=True)
                st.image(blurred2, caption="Second Gaussian Blur", use_column_width=True)

                fig, ax = plt.subplots()
                ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
                ax.set_xlim(0, 255)
                st.pyplot(fig)
        elif processing_option == "Contrast Based Edge Detection":
                contrast_edge, edge_output, average_output = contrast_based_edge_detection(image)
                contrast_edge_colored = plt.cm.hot(contrast_edge)  # Apply 'hot' colormap

                st.image(contrast_edge_colored, caption="Contrast Based Edge Detection", use_column_width=True)
        elif processing_option == "Variance Operator":
            processed = variance_operator(image)
            st.image(processed, caption="Variance Processed Image", use_column_width=True)
            
            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            st.pyplot(fig)

        elif processing_option == "Range Operator":
            processed = range_operator(image)
            st.image(processed, caption="Range Processed Image", use_column_width=True)
            
            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            st.pyplot(fig)
        elif processing_option == "Convolution":
            mask_option = st.sidebar.selectbox(
                "Select Mask for Convolution",
                ["High-pass Filter", "Low-pass Filter"]
            )
            
            # Choose the mask based on the selected option
            if mask_option == "High-pass Filter":
                processed = conv(image, mask_high_pass)
            else:  # Low-pass Filter
                processed = conv(image, mask_low_pass)
            
            st.image(processed, caption=f"Processed Image with {mask_option}", use_column_width=True)
            
            # Display histogram for the processed image
            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            st.pyplot(fig)
        
        elif processing_option == "Median Filter":
            processed = median_filter(image)
            st.image(processed, caption="Median Filter Processed Image", use_column_width=True)
            
            # Display histogram for the processed image
            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            st.pyplot(fig)
        elif processing_option == "Add Images":
            uploaded_file2 = st.file_uploader("Choose a second image file for addition", type=['jpg', 'jpeg', 'png'])
            if uploaded_file2 is not None:
                image2 = np.array(Image.open(uploaded_file2))  # Load second image
                # Convert second image to grayscale
                if len(image2.shape) == 3:
                    image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
                else:
                    image2_gray = image2  # Already grayscale
                
                # Perform image addition
                processed = add(image_gray, image2_gray)
                st.image(processed, caption="Added Image", use_column_width=True)
                
                # Display histogram
                fig, ax = plt.subplots()
                ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
                ax.set_xlim(0, 255)
                st.pyplot(fig)

        elif processing_option == "Subtract Images":
            uploaded_file2 = st.file_uploader("Choose a second image file for subtraction", type=['jpg', 'jpeg', 'png'])
            if uploaded_file2 is not None:
                image2 = np.array(Image.open(uploaded_file2))  # Load second image
                # Convert second image to grayscale
                if len(image2.shape) == 3:
                    image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
                else:
                    image2_gray = image2  # Already grayscale
                
                # Perform image subtraction
                processed = subtract(image_gray, image2_gray)
                st.image(processed, caption="Subtracted Image", use_column_width=True)
                
                # Display histogram
                fig, ax = plt.subplots()
                ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
                ax.set_xlim(0, 255)
                st.pyplot(fig)

        elif processing_option == "Invert Image":
            # Perform image inversion
            processed = invert(image_gray)
            st.image(processed, caption="Inverted Image", use_column_width=True)
            
            # Display histogram
            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            st.pyplot(fig)

        elif processing_option == "Cut and Paste":
            # Take inputs for position and size
            x = st.sidebar.slider("X Position", 0, image_gray.shape[1] - 1, 50)
            y = st.sidebar.slider("Y Position", 0, image_gray.shape[0] - 1, 50)
            w = st.sidebar.slider("Width of Cut Region", 1, image_gray.shape[1] - x, 50)
            h = st.sidebar.slider("Height of Cut Region", 1, image_gray.shape[0] - y, 50)
            position = (x, y)
            size = (w, h)
            
            uploaded_file2 = st.file_uploader("Choose a second image for cut and paste operation", type=['jpg', 'jpeg', 'png'])
            if uploaded_file2 is not None:
                image2 = np.array(Image.open(uploaded_file2))  # Load second image
                # Convert second image to grayscale
                if len(image2.shape) == 3:
                    image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
                else:
                    image2_gray = image2  # Already grayscale
                
                # Perform cut and paste operation
                processed = cut_paste(image_gray, image2_gray, position, size)
                st.image(processed, caption="Cut and Pasted Image", use_column_width=True)
                
                # Display histogram
                fig, ax = plt.subplots()
                ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
                ax.set_xlim(0, 255)
                st.pyplot(fig)
        


if __name__ == "__main__":
    main()


