from app import app, db
from database.models import Product, CartItem, Order, OrderItem
import os

def reset_database():
    with app.app_context():
        # Delete all records from related tables
        OrderItem.query.delete()
        Order.query.delete()
        CartItem.query.delete()
        Product.query.delete()
        
        # Commit the changes
        db.session.commit()
        
        print("Database reset successfully!")

def setup_directories():
    # Create necessary directories
    directories = [
        'static/uploads/products',
        'static/images'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        os.chmod(directory, 0o755)
    
    print("Directories created successfully!")

if __name__ == "__main__":
    setup_directories()
    reset_database() 