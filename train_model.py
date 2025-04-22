import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer, default_data_collator
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

class WasteDataset(Dataset):
    def __init__(self, data_dir, processor, transform=None):
        self.data_dir = data_dir
        self.processor = processor
        self.transform = transform
        self.class_names = [d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        self.images = []
        self.labels = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Manual preprocessing with ViT normalization values
        image = image.resize((224, 224))
        image = np.array(image).transpose(2, 0, 1).astype(np.float32)
        
        # Normalize with ViT mean and std
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        image = (image / 255.0 - mean) / std
        
        return {
            'pixel_values': torch.tensor(image, dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    report = classification_report(labels, predictions, output_dict=True)
    return {
        'accuracy': (predictions == labels).mean(),
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score']
    }

def train_waste_classifier(data_dir, model_save_path, epochs=20, batch_size=32):
    """
    Train a waste classification model using Vision Transformer (ViT)
    
    Parameters:
        data_dir: Directory containing training data in subdirectories
        model_save_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    # Initialize ViT processor and model
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=3,  # recyclable, compostable, general_waste
        ignore_mismatched_sizes=True
    )
    
    # Create datasets
    dataset = WasteDataset(data_dir, processor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        gradient_accumulation_steps=4,  # Accumulate gradients for larger effective batch size
        learning_rate=2e-5,  # Slightly higher learning rate for faster convergence
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(model_save_path)
    processor.save_pretrained(model_save_path)
    
    # Plot training history
    plot_training_history(trainer.state.log_history)
    
    # Create confusion matrix
    plot_confusion_matrix(trainer, val_dataset)
    
    # Generate classification report
    generate_classification_report(trainer, val_dataset)
    
    return model, processor

def plot_training_history(history):
    """Plot training metrics"""
    plt.figure(figsize=(15, 6))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    train_acc = [x['accuracy'] for x in history if 'accuracy' in x]
    val_acc = [x['eval_accuracy'] for x in history if 'eval_accuracy' in x]
    plt.plot(train_acc, 'o-', label='Training Accuracy', linewidth=2)
    plt.plot(val_acc, 'o-', label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    train_loss = [x['loss'] for x in history if 'loss' in x]
    val_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]
    plt.plot(train_loss, 'o-', label='Training Loss', linewidth=2)
    plt.plot(val_loss, 'o-', label='Validation Loss', linewidth=2)
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(trainer, dataset):
    """Plot confusion matrix"""
    predictions = trainer.predict(dataset)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def generate_classification_report(trainer, dataset):
    """Generate classification report"""
    predictions = trainer.predict(dataset)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv('classification_report.csv')
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

def prepare_sample_dataset():
    """Prepare a sample dataset structure for waste classification"""
    dataset_dir = 'sample_dataset'
    categories = ['recyclable', 'compostable', 'general_waste']
    
    os.makedirs(dataset_dir, exist_ok=True)
    for category in categories:
        os.makedirs(os.path.join(dataset_dir, category), exist_ok=True)
    
    print(f"Created sample dataset structure at {dataset_dir}")
    print("In a real application, you would need to populate this with actual images")
    print("For each category (recyclable, compostable, general_waste)")
    
    print("\nRecommendations for dataset collection:")
    print("1. Aim for at least 1000 images per category for good performance")
    print("2. Include various lighting conditions and backgrounds")
    print("3. Capture objects from different angles")
    print("4. Include different types of items within each category")
    print("5. Consider using data augmentation to expand your dataset")
    
    return dataset_dir

if __name__ == "__main__":
    # Prepare dataset structure
    data_dir = prepare_sample_dataset()
    
    # Set the path to save the trained model
    model_save_path = 'models/saved_model/waste_classifier'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Train the model
    model, processor = train_waste_classifier(data_dir, model_save_path)
    
    print("\nModel training complete.")
    print(f"Model saved to {model_save_path}")
    print("You can now use this model for waste classification.")
