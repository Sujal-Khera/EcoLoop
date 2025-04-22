import os
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from models.classifier import WasteClassifier

# Load environment variables
load_dotenv()

# Import services
from services.image_service import process_image
from services.huggingface_service import HuggingFaceService
from services.location_service import find_recycling_centers
from database.db import init_db
from database.models import User, Scan, RecyclingLocation, db, Product, CartItem, Order, OrderItem
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Add a secret key for session management
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Initialize database
init_db(app)

# Initialize the classifier with Hugging Face model
classifier = WasteClassifier("SujalKh/waste-classifier")

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Middleware to check if user is logged in
def login_required(f):
    def wrap(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save the uploaded file temporarily
        temp_path = os.path.join('temp', file.filename)
        os.makedirs('temp', exist_ok=True)
        file.save(temp_path)
        
        # Get prediction
        class_label, predicted_class, confidence = classifier.predict(temp_path)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        return jsonify({
            'class': class_label,
            'confidence': float(confidence)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def classify_image():
    """API endpoint to classify uploaded waste images"""
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
            
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
            
        # Save uploaded image
        filename = secure_filename(image_file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        image_file.save(filepath)
        
        # Classify image directly using file path
        classification, confidence = classifier.predict(filepath)
        
        # Map classification to waste type
        waste_types = {
            0: 'recyclable',
            1: 'compostable',
            2: 'general_waste'
        }
        waste_type = waste_types.get(classification, 'unknown')
        
        # Log the scan in database if user is authenticated
        user_id = session.get('user_id')
        if user_id:
            scan = Scan(
                user_id=user_id,
                image_path=filepath,
                classification=waste_type,
                confidence=float(confidence)
            )
            db.session.add(scan)
            db.session.commit()
            
        # Return classification result
        return jsonify({
            'waste_type': waste_type,
            'confidence': float(confidence),
            'image_id': saved_filename
        })
        
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        return jsonify({'error': 'Classification failed', 'details': str(e)}), 500

@app.route('/api/recycling-centers', methods=['GET'])
def get_recycling_centers():
    """API endpoint to find recycling centers near a location"""
    try:
        # Get parameters
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        waste_type = request.args.get('type', 'recyclable')
        
        if not lat or not lng:
            return jsonify({'error': 'Latitude and longitude are required'}), 400
            
        # Find nearby recycling centers
        centers = find_recycling_centers(lat, lng, waste_type)
        
        return jsonify(centers)
        
    except Exception as e:
        logger.error(f"Location service error: {str(e)}")
        return jsonify({'error': 'Could not find recycling centers', 'details': str(e)}), 500

# Marketplace route: lists available products
@app.route('/marketplace')
def marketplace():
    products = Product.query.all()
    return render_template('marketplace.html', products=products)

# Add product route
@app.route('/add-product', methods=['GET', 'POST'])
@login_required
def add_product():
    if request.method == 'POST':
        try:
            title = request.form.get('title')
            description = request.form.get('description')
            price = float(request.form.get('price'))
            category = request.form.get('category')
            image = request.files.get('image')
            
            if not all([title, description, price, category]):
                flash('Please fill in all fields', 'error')
                return redirect(url_for('add_product'))
            
            # Handle image upload
            image_path = 'images/placeholder.jpg'  # Default image
            if image and image.filename:
                filename = secure_filename(image.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                saved_filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.static_folder, 'uploads', 'products', saved_filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                image.save(filepath)
                image_path = f"static/uploads/products/{saved_filename}"
            
            # Create new product
            product = Product(
                title=title,
                description=description,
                price=price,
                category=category,
                image_path=image_path,
                seller_id=session['user_id'],
                created_at=datetime.utcnow()
            )
            
            db.session.add(product)
            db.session.commit()
            
            flash('Product added successfully!', 'success')
            return redirect(url_for('marketplace'))
            
        except Exception as e:
            logger.error(f"Error adding product: {str(e)}")
            flash('Error adding product. Please try again.', 'error')
            return redirect(url_for('add_product'))
    
    return render_template('add_product.html')

# Product detail route
@app.route('/product/<int:product_id>')
def product_detail(product_id):
    product = Product.query.get_or_404(product_id)
    return render_template('product_detail.html', product=product)

# Cart routes
@app.route('/api/cart/add', methods=['POST'])
@login_required
def add_to_cart():
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        quantity = data.get('quantity', 1)
        
        product = Product.query.get_or_404(product_id)
        
        cart_item = CartItem(
            user_id=session['user_id'],
            product_id=product_id,
            quantity=quantity,
            created_at=datetime.utcnow()
        )
        
        db.session.add(cart_item)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Item added to cart'
        })
        
    except Exception as e:
        logger.error(f"Error adding to cart: {str(e)}")
        return jsonify({'error': 'Failed to add item to cart'}), 500

@app.route('/api/cart/remove/<int:cart_item_id>', methods=['DELETE'])
@login_required
def remove_from_cart(cart_item_id):
    try:
        cart_item = CartItem.query.get_or_404(cart_item_id)
        
        # Verify the cart item belongs to the current user
        if cart_item.user_id != session['user_id']:
            return jsonify({'error': 'Unauthorized'}), 403
            
        db.session.delete(cart_item)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Item removed from cart'
        })
        
    except Exception as e:
        logger.error(f"Error removing cart item: {str(e)}")
        return jsonify({'error': 'Failed to remove item from cart'}), 500

@app.route('/api/cart/update', methods=['POST'])
@login_required
def update_cart():
    try:
        data = request.get_json()
        cart_item_id = data.get('cart_item_id')
        quantity = data.get('quantity')
        
        cart_item = CartItem.query.get_or_404(cart_item_id)
        if cart_item.user_id != session['user_id']:
            return jsonify({'error': 'Unauthorized'}), 403
            
        cart_item.quantity = quantity
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Cart updated'
        })
        
    except Exception as e:
        logger.error(f"Error updating cart: {str(e)}")
        return jsonify({'error': 'Failed to update cart'}), 500

@app.route('/api/cart/items')
@login_required
def get_cart_items():
    try:
        cart_items = CartItem.query.filter_by(user_id=session['user_id'])\
            .join(Product)\
            .all()
            
        return jsonify([{
            'id': item.id,
            'product_id': item.product_id,
            'title': item.product.title,
            'price': item.product.price,
            'quantity': item.quantity,
            'image': url_for('static', filename=item.product.image_path)
        } for item in cart_items])
        
    except Exception as e:
        logger.error(f"Error getting cart items: {str(e)}")
        return jsonify({'error': 'Failed to get cart items'}), 500

@app.route('/checkout')
@login_required
def checkout():
    return render_template('checkout.html')

@app.route('/api/checkout', methods=['POST'])
@login_required
def process_checkout():
    try:
        data = request.get_json()
        
        # Get cart items
        cart_items = CartItem.query.filter_by(user_id=session['user_id'])\
            .join(Product)\
            .all()
            
        if not cart_items:
            return jsonify({'error': 'Cart is empty'}), 400
            
        # Calculate totals
        subtotal = sum(item.product.price * item.quantity for item in cart_items)
        shipping_cost = 5.00
        total = subtotal + shipping_cost
        
        # Create order
        order = Order(
            user_id=session['user_id'],
            status='confirmed',
            subtotal=subtotal,
            shipping_cost=shipping_cost,
            total=total,
            shipping_address=data.get('address'),
            shipping_city=data.get('city'),
            shipping_state=data.get('state'),
            shipping_zip=data.get('zip'),
            customer_name=f"{data.get('firstName')} {data.get('lastName')}",
            customer_email=data.get('email')
        )
        
        db.session.add(order)
        db.session.flush()  # Get order ID without committing
        
        # Create order items
        for cart_item in cart_items:
            order_item = OrderItem(
                order_id=order.id,
                product_id=cart_item.product_id,
                quantity=cart_item.quantity,
                price=cart_item.product.price
            )
            db.session.add(order_item)
            
        # Clear cart
        for item in cart_items:
            db.session.delete(item)
            
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'order_id': order.id
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error processing checkout: {str(e)}")
        return jsonify({'error': 'Failed to process checkout'}), 500

@app.route('/order-confirmation/<int:order_id>')
@login_required
def order_confirmation(order_id):
    order = Order.query.get_or_404(order_id)
    
    # Ensure user owns this order
    if order.user_id != session['user_id']:
        abort(403)
        
    return render_template('order_confirmation.html', order=order)

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm-password')

        # Validate form data
        if not all([username, email, password, confirm_password]):
            flash('Please fill in all fields.', 'error')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')

        # Check for existing username or email
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            if existing_user.username == username:
                flash('Username already exists. Please choose a different username.', 'error')
            if existing_user.email == email:
                flash('Email already registered. Please use a different email.', 'error')
            return render_template('register.html')

        # Create new user
        hashed_password = generate_password_hash(password)
        new_user = User(
            username=username,
            email=email,
            password_hash=hashed_password,
            created_at=datetime.utcnow()
        )
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Validate form data
        if not all([username, password]):
            flash('Please fill in all fields.', 'error')
            return render_template('login.html')

        # Check if user exists and password matches
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            # Update last login
            user.last_login = datetime.utcnow()
            db.session.commit()
            # Set user session
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')
            return render_template('login.html')

    return render_template('login.html')

# Dashboard route (combined for simplicity)
@app.route('/dashboard')
@login_required
def dashboard():
    user = User.query.get(session['user_id'])
    return render_template('dashboard.html', user=user)

# Seller dashboard (protected)
@app.route('/seller/dashboard')
@login_required
def seller_dashboard():
    return render_template('seller_dashboard.html')

# Buyer dashboard (protected)
@app.route('/buyer/dashboard')
@login_required
def buyer_dashboard():
    return render_template('buyer_dashboard.html')

# Logout route
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    return render_template('contact.html')

@app.route('/classification-dashboard')
@login_required
def classification_dashboard():
    """Render the waste classification dashboard"""
    return render_template('classification_dashboard.html')

@app.route('/api/save-scan', methods=['POST'])
@login_required
def save_scan():
    """Save a waste classification scan to the database"""
    try:
        data = request.get_json()
        user_id = session.get('user_id')
        
        # Create new scan record
        scan = Scan(
            user_id=user_id,
            image_path=data.get('image_path'),
            classification=data.get('waste_type'),
            confidence=float(data.get('confidence').replace('%', '')),
            created_at=datetime.utcnow()
        )
        
        db.session.add(scan)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'scan_id': scan.id,
            'waste_type': scan.classification,
            'confidence': scan.confidence,
            'created_at': scan.created_at.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error saving scan: {str(e)}")
        return jsonify({'error': 'Failed to save scan'}), 500

@app.route('/api/user-stats')
@login_required
def get_user_stats():
    """Get user statistics including total scans and recycled items"""
    try:
        user_id = session.get('user_id')
        
        # Get total scans
        total_scans = Scan.query.filter_by(user_id=user_id).count()
        
        # Get recycled items (scans classified as recyclable)
        recycled_items = Scan.query.filter_by(
            user_id=user_id,
            classification='recyclable'
        ).count()
        
        return jsonify({
            'total_scans': total_scans,
            'recycled_items': recycled_items
        })
        
    except Exception as e:
        logger.error(f"Error getting user stats: {str(e)}")
        return jsonify({'error': 'Failed to get user statistics'}), 500

@app.route('/api/products')
def get_products():
    """Get all available products"""
    try:
        products = Product.query.all()
        return jsonify([{
            'id': product.id,
            'title': product.title,
            'description': product.description,
            'price': product.price,
            'category': product.category,
            'image_path': product.image_path,
            'seller_id': product.seller_id,
            'created_at': product.created_at.isoformat()
        } for product in products])
    except Exception as e:
        logger.error(f"Error fetching products: {str(e)}")
        return jsonify({'error': 'Failed to fetch products'}), 500

@app.route('/api/scan-history')
@login_required
def get_scan_history():
    """Get user's recent scan history"""
    try:
        user_id = session.get('user_id')
        limit = request.args.get('limit', 10, type=int)
        
        # Get recent scans
        scans = Scan.query.filter_by(user_id=user_id)\
            .order_by(Scan.created_at.desc())\
            .limit(limit)\
            .all()
        
        return jsonify([{
            'id': scan.id,
            'waste_type': scan.classification,
            'confidence': scan.confidence,
            'created_at': scan.created_at.isoformat(),
            'image_path': scan.image_path
        } for scan in scans])
        
    except Exception as e:
        logger.error(f"Error getting scan history: {str(e)}")
        return jsonify({'error': 'Failed to get scan history'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)