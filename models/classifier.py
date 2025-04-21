import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class WasteClassifier:
    """Waste classification model using MobileNetV2 with transfer learning"""
    
    def __init__(self, model_path=None):
        """Initialize the model, loading a pre-trained model if available"""
        self.img_size = (224, 224)  # MobileNetV2 default input size
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # If no model is available, create a new one
            self._create_model()
            
    def _create_model(self):
        """Create and compile the model architecture with advanced features"""
        # Base MobileNetV2 model (pre-trained on ImageNet)
        base_model = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Initially freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Add custom classification layers with dropout and batch normalization
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        predictions = Dense(3, activation='softmax')(x)  # 3 classes: recyclable, compostable, general waste
        
        # Final model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Created new model. This model needs to be trained before use.")
        
    def load_model(self, model_path):
        """Load a saved model from disk"""
        try:
            self.model = load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self._create_model()
            
    def save_model(self, model_path):
        """Save the model to disk"""
        if self.model:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
            
    def train(self, train_data, validation_data, epochs=10, batch_size=32, callbacks=None):
        """
        Train the model on waste image data with advanced training strategy

        Parameters:
        train_data: Training data generator or tuple (x_train, y_train)
        validation_data: Validation data generator or tuple (x_val, y_val)
        epochs: Number of training epochs
        batch_size: Batch size for training
        callbacks: List of tf.keras.callbacks to use during training.
        """
        if not self.model:
            self._create_model()

        # Progressive unfreezing for better fine-tuning
        # First train only the custom layers
        for layer in self.model.layers[-6:]:
            layer.trainable = True

        # Initial compilation with higher learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        # Train for a few epochs
        initial_history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs//2,
            batch_size=batch_size,
            callbacks=callbacks
        )

        # Unfreeze more layers for fine-tuning
        for layer in self.model.layers[-20:]:
            layer.trainable = True

        # Recompile with a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        # Train the model
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks  # ✅ Fixed: Added callbacks argument
        )

        return history
            
    def preprocess_image(self, image_path):
        """
        Preprocess an image for model prediction
        
        Parameters:
        image_path: Path to the image file
        
        Returns:
        Preprocessed image array ready for model input
        """
        img = load_img(image_path, target_size=self.img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
        
    def predict(self, image):
        """
        Classify an image
        
        Parameters:
        image: Preprocessed image array or path to image file
        
        Returns:
        predicted_class: Class index (0=recyclable, 1=compostable, 2=general_waste)
        confidence: Confidence score for the prediction
        """
        if not self.model:
            raise ValueError("Model not loaded. Please load or train a model first.")
            
        # Process image if a file path is provided
        if isinstance(image, str):
            image = self.preprocess_image(image)
            
        # Get predictions
        predictions = self.model.predict(image)
        
        # Get class with highest confidence
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        return predicted_class, confidence
        
    def get_training_data_generator(self, data_dir, batch_size=32):
        """
        Create a data generator for training from a directory structure
        Expects a structure like:
            data_dir/
                recyclable/
                    image1.jpg
                    image2.jpg
                    ...
                compostable/
                    image1.jpg
                    ...
                general_waste/
                    image1.jpg
                    ...
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Enhanced data augmentation for training
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            channel_shift_range=50.0,
            fill_mode='reflect',
            validation_split=0.2  # Use 20% for validation
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Validation generator
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, validation_generator
