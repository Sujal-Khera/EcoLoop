import os

class Config:
    """Base configuration class for the application"""
    # Flask configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-for-development-only')
    DEBUG = os.getenv('DEBUG', 'True').lower() in ('true', '1', 't')
    
    # Upload configuration
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///waste_classification.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # API Keys
    GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', '')
    
    # Model configuration
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/saved_model/waste_classifier.h5')
    
class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///test.db'
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.getenv('SECRET_KEY')  # Must be set in production
    
    # In production, consider using a more robust database
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
    
    # In production, consider using cloud storage for uploads
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/app/uploads')
    
# Configuration dictionary - could be used to select config based on environment
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}