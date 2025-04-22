import os
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms

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
        img = Image.open(image_path).convert('RGB')
        
        # Define preprocessing transforms
        preprocess = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Apply preprocessing
        preprocessed_img = preprocess(img)
        
        # Add batch dimension
        preprocessed_img = preprocessed_img.unsqueeze(0)
        
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
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Save the enhanced image
        if output_path is None:
            output_path = image_path
        
        cv2.imwrite(output_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
        return output_path
        
    except Exception as e:
        print(f"Error enhancing image: {str(e)}")
        raise

def extract_image_features(image_path):
    """
    Extract features from an image for classification
    
    Parameters:
    image_path: Path to the image file
    
    Returns:
    Extracted features as a numpy array
    """
    try:
        # Process the image
        preprocessed_img = process_image(image_path)
        
        # Convert to numpy array
        features = preprocessed_img.numpy()
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        raise