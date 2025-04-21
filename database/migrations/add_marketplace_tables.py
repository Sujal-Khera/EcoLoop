from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

def upgrade(db: SQLAlchemy):
    # Create Product table
    db.create_table('products',
        db.Column('id', db.Integer, primary_key=True),
        db.Column('title', db.String(100), nullable=False),
        db.Column('description', db.Text, nullable=False),
        db.Column('price', db.Float, nullable=False),
        db.Column('category', db.String(50), nullable=False),
        db.Column('image_path', db.String(255), nullable=False),
        db.Column('seller_id', db.Integer, db.ForeignKey('users.id'), nullable=False),
        db.Column('created_at', db.DateTime, default=datetime.utcnow)
    )
    
    # Create CartItem table
    db.create_table('cart_items',
        db.Column('id', db.Integer, primary_key=True),
        db.Column('user_id', db.Integer, db.ForeignKey('users.id'), nullable=False),
        db.Column('product_id', db.Integer, db.ForeignKey('products.id'), nullable=False),
        db.Column('quantity', db.Integer, default=1),
        db.Column('created_at', db.DateTime, default=datetime.utcnow)
    )
    
    # Add cart_items relationship to User
    db.add_column('users', 'cart_items', db.relationship('CartItem', backref='user', lazy=True))

def downgrade(db: SQLAlchemy):
    # Drop CartItem table
    db.drop_table('cart_items')
    
    # Drop Product table
    db.drop_table('products')
    
    # Remove cart_items relationship from User
    db.drop_column('users', 'cart_items') 