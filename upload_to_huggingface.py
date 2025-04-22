import os
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from huggingface_hub import HfApi, create_repo, Repository

def convert_to_huggingface(model_path, output_dir):
    """Convert the model to Hugging Face format"""
    print("Loading model...")
    try:
        # Initialize ViT model
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=3,  # recyclable, compostable, general_waste
            ignore_mismatched_sizes=True
        )
        
        # Load weights if they exist
        if os.path.exists(model_path):
            print("Loading weights...")
            model.load_state_dict(torch.load(model_path))
        
        print("Model loaded successfully")
        print("\nPreparing model for Hugging Face...")
        
        # Save model
        model.save_pretrained(output_dir)
        
        # Save processor
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        processor.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
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