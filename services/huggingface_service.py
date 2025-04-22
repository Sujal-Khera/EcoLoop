import os
import requests
import json
from PIL import Image
import io
import base64

class HuggingFaceService:
    def __init__(self):
        self.api_url = os.getenv('HUGGINGFACE_API_URL')
        self.api_token = os.getenv('HUGGINGFACE_API_TOKEN')
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

    def preprocess_image(self, image_path):
        """Preprocess image for Hugging Face API"""
        try:
            # Open and resize image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if needed (optional, depends on model requirements)
                img = img.resize((224, 224))
                
                # Convert to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Convert to base64
                return base64.b64encode(img_byte_arr).decode('utf-8')
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            raise

    def predict(self, image_path):
        """Make prediction using Hugging Face API"""
        try:
            # Preprocess image
            image_data = self.preprocess_image(image_path)
            
            # Prepare payload
            payload = {
                "inputs": image_data
            }
            
            # Make API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload
            )
            
            # Parse response
            if response.status_code == 200:
                result = response.json()
                
                # Map the prediction to waste types
                waste_types = {
                    'recyclable': 0,
                    'compostable': 1,
                    'general_waste': 2
                }
                
                # Get the highest confidence prediction
                prediction = result[0]
                label = prediction['label']
                confidence = prediction['score']
                
                # Map label to class index
                class_index = waste_types.get(label, 2)  # Default to general_waste if unknown
                
                return class_index, confidence
            else:
                print(f"API Response: {response.text}")  # Add this line for debugging
                raise Exception(f"API request failed with status code {response.status_code}")
                
        except Exception as e:
            print(f"Error in Hugging Face prediction: {str(e)}")
            raise 