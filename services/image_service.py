import os
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def process_image(image_path, target_size=(224, 224)):
    """
    Process an image for classification
    
    Parameters:
    image_path: Path to the image file
    target_size: Target dimensions for resizing
    
    Returns:
    Preprocessed image ready for model prediction
    """
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Resize image to target size
        img = img.resize(target_size)
        
        # Convert to array
        img_array = img_to_array(img)
        
        # Expand dimensions for batch processing
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for MobileNetV2
        preprocessed_img = preprocess_input(img_array)
        
        return preprocessed_img
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise
        
def enhance_image_quality(image_path, output_path=None):
    """
    Enhance image quality for better classification results
    
    Parameters:
    image_path: Path to the input image
    output_path: Path to save the enhanced image (if None, overwrites original)
    
    Returns:
    Path to the enhanced image
    """
    try:
        # Read the image using OpenCV
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Failed to load image from {image_path}")
            
        # Apply basic image enhancements
        
        # 1. Resize if too large or too small
        height, width = img.shape[:2]
        max_dimension = 1200
        min_dimension = 300
        
        if max(height, width) > max_dimension:
            scale_factor = max_dimension / max(height, width)
            img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        elif min(height, width) < min_dimension:
            scale_factor = min_dimension / min(height, width)
            img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            
        # 2. Convert to RGB (from BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 3. Apply slight contrast enhancement
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # 4. Reduce noise if necessary
        denoised = cv2.fastNlMeansDenoisingColored(enhanced_img, None, 10, 10, 7, 21)
        
        # Convert back to BGR for saving with OpenCV
        enhanced_bgr = cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR)
        
        # Save or overwrite the image
        if output_path is None:
            output_path = image_path
            
        cv2.imwrite(output_path, enhanced_bgr)
        
        return output_path
    
    except Exception as e:
        print(f"Error enhancing image: {str(e)}")
        raise
        
def extract_image_features(image_path):
    """
    Extract basic features from an image for additional analysis
    
    Parameters:
    image_path: Path to the image
    
    Returns:
    Dictionary of image features (colors, brightness, etc.)
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Failed to load image from {image_path}")
            
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get dimensions
        height, width = img.shape[:2]
        
        # Calculate average color
        avg_color_per_row = np.average(img_rgb, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        
        # Calculate brightness
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.average(gray)
        
        # Detect edges (can help identify if it's a clear image of an object)
        edges = cv2.Canny(gray, 100, 200)
        edge_percentage = np.count_nonzero(edges) / (height * width) * 100
        
        # Return features
        return {
            'dimensions': (width, height),
            'avg_color_rgb': avg_color.astype(int).tolist(),
            'brightness': brightness,
            'edge_percentage': edge_percentage
        }
    
    except Exception as e:
        print(f"Error extracting image features: {str(e)}")
        raise