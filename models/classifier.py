import os
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import numpy as np

class WasteClassifier:
    """Waste classification model using Vision Transformer (ViT)"""
    
    def __init__(self, model_path=None):
        """
        Initialize the model, loading a pre-trained model if available
        
        Parameters:
            model_path: Path to a saved model directory or Hugging Face repository name
        """
        self.img_size = (224, 224)  # ViT input size
        self.model = None
        self.processor = None
        self.class_labels = ['compostable', 'general_waste', 'recyclable']
        
        if model_path:
            self.load_model(model_path)
        else:
            # Default to loading from Hugging Face Hub
            self.load_model("SujalKh/waste-classifier")
    
    def _load_converted_model(self):
        """Load the converted Hugging Face model"""
        converted_model_path = 'models/saved_model/huggingface_model'
        if os.path.exists(converted_model_path):
            self.load_model(converted_model_path)
        else:
            # If no converted model is available, create a new one
            self._create_model()
    
    def _create_model(self):
        """Create and configure the model architecture"""
        # Initialize ViT processor and model
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=3,  # recyclable, compostable, general_waste
            ignore_mismatched_sizes=True
        )
        
        print(f"Created new model with 3 classes. This model needs to be trained before use.")
    
    def load_model(self, model_path):
        """Load a saved model from disk or Hugging Face Hub"""
        try:
            # Check if model_path is a Hugging Face repository
            if "/" in model_path and not os.path.exists(model_path):
                print(f"Loading model from Hugging Face Hub: {model_path}")
                self.processor = ViTImageProcessor.from_pretrained(model_path)
                self.model = ViTForImageClassification.from_pretrained(model_path)
            else:
                # Load from local path
                print(f"Loading model from local path: {model_path}")
                self.processor = ViTImageProcessor.from_pretrained(model_path)
                self.model = ViTForImageClassification.from_pretrained(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e
    
    def save_model(self, model_path):
        """Save the model to disk"""
        if self.model and self.processor:
            # Ensure directory exists
            os.makedirs(model_path, exist_ok=True)
            self.model.save_pretrained(model_path)
            self.processor.save_pretrained(model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save. Create or train a model first.")
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for model prediction
        
        Parameters:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor ready for model input
        """
        # Open and convert image
        image = Image.open(image_path).convert('RGB')
        
        # Process with ViT processor
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs
    
    def predict(self, image):
        """
        Classify an image
        
        Parameters:
            image: Path to image file or preprocessed image tensor
            
        Returns:
            class_label: String label (recyclable, compostable, general_waste)
            predicted_class: Class index (0=recyclable, 1=compostable, 2=general_waste)
            confidence: Confidence score for the prediction
        """
        if not self.model or not self.processor:
            raise ValueError("Model not loaded. Please load or train a model first.")
        
        try:
            # Process image if a file path is provided
            if isinstance(image, str):
                inputs = self.preprocess_image(image)
            else:
                inputs = image
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits.softmax(dim=1)
            
            # Get class with highest confidence
            predicted_class = torch.argmax(predictions[0]).item()
            confidence = float(predictions[0][predicted_class])
            class_label = self.class_labels[predicted_class]
            
            return class_label, predicted_class, confidence
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise e
    
    def evaluate(self, test_data):
        """
        Evaluate the model on test data
        
        Parameters:
            test_data: Test data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.model:
            raise ValueError("Model not loaded. Please load or train a model first.")
        
        self.model.eval()
        total = 0
        correct = 0
        
        with torch.no_grad():
            for batch in test_data:
                inputs = batch['pixel_values']
                labels = batch['labels']
                
                outputs = self.model(inputs)
                predictions = outputs.logits.argmax(dim=1)
                
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        
        accuracy = correct / total
        print(f"Test accuracy: {accuracy:.4f}")
        return {'accuracy': accuracy}
