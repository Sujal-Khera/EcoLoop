from huggingface_hub import HfApi, create_repo
from transformers import ViTForImageClassification, ViTImageProcessor
import os
from dotenv import load_dotenv

def upload_to_huggingface(local_model_path, repo_name, token):
    """
    Upload model to Hugging Face Hub
    
    Parameters:
        local_model_path: Path to the local model directory
        repo_name: Name for the Hugging Face repository
        token: Hugging Face API token
    """
    # Initialize Hugging Face API
    api = HfApi()
    
    try:
        # Try to create repository (will succeed if it doesn't exist)
        try:
            print(f"Creating repository {repo_name}...")
            create_repo(repo_name, token=token, private=True)
        except Exception as e:
            if "409" in str(e):
                print(f"Repository {repo_name} already exists. Continuing with upload...")
            else:
                raise e
        
        # Upload model and processor files
        print("Uploading model files...")
        api.upload_folder(
            folder_path=local_model_path,
            repo_id=repo_name,
            token=token
        )
        
        print(f"Model successfully uploaded to: https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"Error uploading model: {str(e)}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Get Hugging Face token from environment variable
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("Please set your HUGGINGFACE_TOKEN environment variable")
        exit(1)
    
    # Get Hugging Face username
    username = input("Enter your Hugging Face username: ").strip()
    if not username:
        print("Username cannot be empty")
        exit(1)
    
    # Set paths and names
    local_model_path = "models/saved_model/huggingface_model"
    repo_name = f"{username}/waste-classifier"
    
    # Upload model
    upload_to_huggingface(local_model_path, repo_name, token) 