from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    """User model for storing user account information"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    scans = db.relationship('Scan', backref='user', lazy=True)
    cart_items = db.relationship('CartItem', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Scan(db.Model):
    """Model for storing waste image scans and classifications"""
    __tablename__ = 'scans'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    waste_type = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Scan {self.id} - {self.waste_type}>'
        
class RecyclingLocation(db.Model):
    """Model for storing recycling center locations"""
    __tablename__ = 'recycling_locations'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    address = db.Column(db.String(255), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    waste_types = db.Column(db.String(255))
    accepts = db.Column(db.String(256), nullable=False)  # Comma-separated list of accepted waste types
    rating = db.Column(db.Float, nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    website = db.Column(db.String(256), nullable=True)
    hours = db.Column(db.String(512), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<RecyclingLocation {self.name}>'
        
class Product(db.Model):
    """Model for storing marketplace products"""
    __tablename__ = 'products'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    price = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    seller_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    cart_items = db.relationship('CartItem', backref='product', lazy=True)

class CartItem(db.Model):
    """Model for storing shopping cart items"""
    __tablename__ = 'cart_items'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    quantity = db.Column(db.Integer, default=1)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Order(db.Model):
    """Model for storing orders"""
    __tablename__ = 'orders'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='pending')
    subtotal = db.Column(db.Float, nullable=False)
    shipping_cost = db.Column(db.Float, nullable=False, default=5.00)
    total = db.Column(db.Float, nullable=False)
    
    # Shipping information
    shipping_address = db.Column(db.String(255), nullable=False)
    shipping_city = db.Column(db.String(100), nullable=False)
    shipping_state = db.Column(db.String(50), nullable=False)
    shipping_zip = db.Column(db.String(20), nullable=False)
    
    # Customer information
    customer_name = db.Column(db.String(100), nullable=False)
    customer_email = db.Column(db.String(120), nullable=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    items = db.relationship('OrderItem', backref='order', lazy=True)
    user = db.relationship('User', backref='orders')

class OrderItem(db.Model):
    """Model for storing order items"""
    __tablename__ = 'order_items'
    
    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey('orders.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('products.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)  # Price at time of purchase
    
    product = db.relationship('Product')

# Initialize database
def init_db(app):
    """Initialize the database with the application"""
    db.init_app(app)
    
    # Create all tables if they don't exist
    with app.app_context():
        db.create_all()