import os
import tensorflow as tf
from transformers import AutoFeatureExtractor
import numpy as np
from huggingface_hub import HfApi, create_repo, Repository
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Input

def convert_to_huggingface(model_path, output_dir):
    """Convert the MobileNetV2-based model to Hugging Face format"""
    print("Loading model...")
    try:
        # Create a new model with the same architecture
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        predictions = Dense(3, activation='softmax')(x)
        
        # Create the model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Load weights
        print("Loading weights...")
        model.load_weights(model_path)
        print("Model loaded successfully")
        model.summary()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
    
    print("\nPreparing model for Hugging Face...")
    try:
        # Save model in SavedModel format
        tf.saved_model.save(model, output_dir)
        print("Model saved in SavedModel format")
        
        # Create model card
        with open(os.path.join(output_dir, "README.md"), "w") as f:
            f.write("""---
language: en
license: mit
tags:
- waste-classification
- image-classification
- tensorflow
datasets:
- custom
---

# Waste Classification Model

This model classifies waste items into three categories:
- Recyclable
- Compostable
- General Waste

## Model Description

The model is based on MobileNetV2 architecture and has been fine-tuned on a custom waste classification dataset.

### Input

The model expects RGB images of size 224x224 pixels.

### Output

The model outputs probabilities for three classes:
- 0: Recyclable
- 1: Compostable
- 2: General Waste

## Usage

```python
from transformers import pipeline

classifier = pipeline("image-classification", model="SujalKh/waste-classifier")
result = classifier("path/to/image.jpg")
print(result)
```
""")
        print("Created model card")
        
    except Exception as e:
        print(f"Error preparing model: {str(e)}")
        raise

def upload_to_hub(model_dir, repo_name, token):
    """Upload the converted model to Hugging Face Hub"""
    if not token:
        raise ValueError("HUGGINGFACE_API_TOKEN environment variable is not set")
    
    api = HfApi()
    
    print(f"\nCreating/accessing repository {repo_name}...")
    try:
        create_repo(repo_name, token=token, exist_ok=True)
        print("Repository ready")
    except Exception as e:
        print(f"Error with repository: {str(e)}")
        raise
    
    print("\nUploading to Hugging Face...")
    try:
        # Clone the repo
        repo = Repository(
            local_dir=model_dir,
            clone_from=repo_name,
            use_auth_token=token
        )
        
        # Add model files
        repo.git_add()
        repo.git_commit("Add model files")
        repo.git_push()
        print("Upload completed successfully")
    except Exception as e:
        print(f"Error during upload: {str(e)}")
        raise

if __name__ == "__main__":
    # Your model path
    model_path = "models/saved_model/waste_classifier.h5"
    
    # Output directory for converted model
    output_dir = "converted_model"
    os.makedirs(output_dir, exist_ok=True)
    
    # Hugging Face repository name
    repo_name = "SujalKh/waste-classifier"
    
    # Your Hugging Face token
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        raise ValueError("Please set the HUGGINGFACE_API_TOKEN environment variable")
    
    print("Starting model conversion process...")
    convert_to_huggingface(model_path, output_dir)
    
    print("\nStarting upload process...")
    upload_to_hub(output_dir, repo_name, token)
    
    print("\nProcess completed successfully!") 