import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.classifier import WasteClassifier
import matplotlib.pyplot as plt

def train_waste_classifier(data_dir, model_save_path, epochs=50, batch_size=16):
    """
    Train a waste classification model using transfer learning with advanced techniques
    
    Parameters:
    data_dir: Directory containing training data in subdirectories
    model_save_path: Path to save the trained model
    epochs: Number of training epochs (default increased for better learning)
    batch_size: Batch size for training (reduced for better generalization)
    
    Returns:
    Trained model and training history
    """
    # Create and configure the classifier
    classifier = WasteClassifier()
    
    # Get training and validation data generators
    train_generator, validation_generator = classifier.get_training_data_generator(
        data_dir,
        batch_size=batch_size
    )
    
    # Enhanced callbacks for better training control and monitoring
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            min_delta=1e-4
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            cooldown=2
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            update_freq='epoch'
        )
    ]
    
    # Train the model
    history = classifier.train(
        train_generator,
        validation_generator,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # Save the model
    classifier.save_model(model_save_path)
    
    # Plot training history
    plot_training_history(history)
    
    return classifier, history

def plot_training_history(history):
    """Plot detailed training metrics including accuracy, loss, precision, and recall"""
    # Create a 2x2 subplot for detailed metrics
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot precision
    plt.subplot(2, 2, 3)
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Model Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    # Plot recall
    plt.subplot(2, 2, 4)
    plt.plot(history.history['recall'], label='Training Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Model Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def prepare_sample_dataset():
    """
    Prepare a small sample dataset structure for demonstration purposes
    
    In a real application, you would collect and organize a large dataset
    of waste images across different categories.
    """
    # Create directory structure for the dataset
    dataset_dir = 'sample_dataset'
    categories = ['recyclable', 'compostable', 'general_waste']
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    for category in categories:
        os.makedirs(os.path.join(dataset_dir, category), exist_ok=True)
        
    print(f"Created sample dataset structure at {dataset_dir}")
    print("In a real application, you would need to populate this with actual images")
    print("For each category (recyclable, compostable, general_waste)")
    
    # Guidance on dataset collection
    print("\nRecommendations for dataset collection:")
    print("1. Aim for at least 500 images per category for basic performance")
    print("2. Include various lighting conditions and backgrounds")
    print("3. Capture objects from different angles")
    print("4. Include different types of items within each category")
    print("5. Consider using data augmentation to expand your dataset")
    
    return dataset_dir

if __name__ == "__main__":
    # Prepare dataset structure
    data_dir = prepare_sample_dataset()
    
    # In a real scenario, you would collect and organize your images
    # into this directory structure first
    
    # Set the path to save the trained model
    model_save_path = 'models/saved_model/waste_classifier.h5'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Train the model when dataset is ready
    # Uncomment the following line when you have collected your dataset
    classifier, history = train_waste_classifier(data_dir, model_save_path)
    
    print("\nNote: Model training code is ready but commented out.")
    print("Uncomment the training line after collecting your dataset.")