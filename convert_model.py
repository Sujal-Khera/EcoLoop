import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import numpy as np
import h5py

def convert_tf_to_huggingface(tf_model_path, hf_model_path):
    """
    Convert TensorFlow model to Hugging Face format
    
    Parameters:
        tf_model_path: Path to the TensorFlow .h5 model
        hf_model_path: Path to save the Hugging Face model
    """
    # Load TensorFlow model weights directly
    print("Loading TensorFlow model weights...")
    with h5py.File(tf_model_path, 'r') as f:
        # Get the model weights
        weights = {}
        for layer_name in f['model_weights']:
            layer = f['model_weights'][layer_name]
            for weight_name in layer:
                weights[f"{layer_name}/{weight_name}"] = np.array(layer[weight_name])
    
    # Initialize Hugging Face model
    print("Initializing Hugging Face model...")
    hf_model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=3,  # recyclable, compostable, general_waste
        ignore_mismatched_sizes=True
    )
    
    # Convert weights
    print("Converting weights...")
    state_dict = {}
    
    # Map TensorFlow weights to Hugging Face model
    for tf_name, weight in weights.items():
        if 'dense' in tf_name:
            if 'kernel' in tf_name:
                # Convert dense layer weights
                weight = np.transpose(weight)  # Transpose to match PyTorch format
                state_dict['classifier.weight'] = torch.from_numpy(weight)
            elif 'bias' in tf_name:
                # Convert dense layer biases
                state_dict['classifier.bias'] = torch.from_numpy(weight)
    
    # Load state dict into Hugging Face model
    hf_model.load_state_dict(state_dict, strict=False)
    
    # Save Hugging Face model
    print(f"Saving Hugging Face model to {hf_model_path}...")
    hf_model.save_pretrained(hf_model_path)
    
    # Save processor
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    processor.save_pretrained(hf_model_path)
    
    print("Conversion complete!")

if __name__ == "__main__":
    # Paths
    tf_model_path = "models/saved_model/waste_classifier.h5"
    hf_model_path = "models/saved_model/huggingface_model"
    
    # Convert model
    convert_tf_to_huggingface(tf_model_path, hf_model_path) 