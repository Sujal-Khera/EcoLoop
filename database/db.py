from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
import os
from .models import db

def init_db(app):
    """
    Initialize the database with the Flask application
    
    Parameters:
    app: Flask application instance
    """
    # Configure database URI
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
        'DATABASE_URL', 
        'sqlite:///waste_classification.db'
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize the app with the extension
    db.init_app(app)
    
    # Create all tables
    with app.app_context():
        db.create_all()
        
def get_db_session(app):
    """
    Get a database session for use outside of request context
    
    Parameters:
    app: Flask application instance
    
    Returns:
    SQLAlchemy session
    """
    engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
    session_factory = sessionmaker(bind=engine)
    return scoped_session(session_factory)