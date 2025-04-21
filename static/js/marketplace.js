document.addEventListener('DOMContentLoaded', function() {
  const searchInput = document.getElementById('searchInput');
  const filterSelect = document.getElementById('filterSelect');
  const sortSelect = document.getElementById('sortSelect');
  const productGrid = document.querySelector('.row.g-4');
  const cartItems = document.getElementById('cartItems');
  const cartTotal = document.getElementById('cartTotal');
  const checkoutBtn = document.getElementById('checkoutBtn');
  const cartButton = document.getElementById('cartButton');
  const cartSidebar = document.getElementById('cartSidebar');
  const closeCart = document.getElementById('closeCart');
  const cartCount = document.getElementById('cartCount');

  let products = [];
  let cart = JSON.parse(localStorage.getItem('cart')) || [];
  let debounceTimer;

  // Fetch products from server
  async function fetchProducts() {
    let retryCount = 0;
    const maxRetries = 3;
    const baseDelay = 1000; // 1 second base delay

    // Show loading state
    productGrid.innerHTML = `
      <div class="col-12 text-center">
        <div class="spinner-border text-primary" role="status">
          <span class="visually-hidden">Loading...</span>
        </div>
        <p class="mt-2">Loading products...</p>
      </div>`;

    while (retryCount < maxRetries) {
      try {
        const response = await fetch('/api/products');
        if (!response.ok) {
          throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        }
        const data = await response.json();
        if (!Array.isArray(data)) {
          throw new Error('Invalid response format');
        }
        products = data;
        applyFiltersAndSort();
        return;
      } catch (error) {
        console.error(`Attempt ${retryCount + 1} failed:`, error);
        retryCount++;
        
        if (retryCount === maxRetries) {
          productGrid.innerHTML = `
            <div class="col-12 text-center">
              <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle me-2"></i>
                <p class="mb-2">Unable to load products. ${error.message}</p>
                <button class="btn btn-primary" onclick="fetchProducts()">
                  <i class="fas fa-sync-alt me-2"></i>Try Again
                </button>
              </div>
            </div>`;
        } else {
          // Exponential backoff
          const delay = baseDelay * Math.pow(2, retryCount - 1);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }
  }

  // Debounced search function
  function debounceSearch(func, delay) {
    return function() {
      const context = this;
      const args = arguments;
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => func.apply(context, args), delay);
    };
  }

  // Filter products based on search term and category
  function filterProducts(products, searchTerm, category) {
    return products.filter(product => {
      const matchesSearch = product.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          product.description?.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesCategory = category === 'all' || product.category === category;
      return matchesSearch && matchesCategory;
    });
  }

  // Sort products based on selected criteria
  function sortProducts(products, sortCriteria) {
    const sortedProducts = [...products];
    switch (sortCriteria) {
      case 'price-low':
        return sortedProducts.sort((a, b) => a.price - b.price);
      case 'price-high':
        return sortedProducts.sort((a, b) => b.price - a.price);
      case 'name':
        return sortedProducts.sort((a, b) => a.title.localeCompare(b.title));
      default:
        return sortedProducts;
    }
  }

  // Apply filters and sort, then update display
  function applyFiltersAndSort() {
    const searchTerm = searchInput.value;
    const category = filterSelect.value;
    const sortCriteria = sortSelect.value;

    let filteredProducts = filterProducts(products, searchTerm, category);
    filteredProducts = sortProducts(filteredProducts, sortCriteria);
    updateProductDisplay(filteredProducts);
  }

  // Update product display in the grid
  function updateProductDisplay(productsToShow) {
    productGrid.innerHTML = productsToShow.length ? 
      productsToShow.map(product => `
        <div class="col-12 col-sm-6 col-md-4 col-lg-3">
          <div class="product-card h-100">
            <div class="product-image-container">
              <img src="${product.image_path}" 
                   alt="${product.title}" 
                   class="product-image"
                   onerror="handleImageError(this)">
            </div>
            <div class="product-info">
              <h5 class="product-title">${product.title}</h5>
              <p class="product-category mb-2">
                <i class="fas fa-tag me-2"></i>${product.category}
              </p>
              <p class="product-price mb-3">$${product.price.toFixed(2)}</p>
              <div class="d-flex gap-2">
                <a href="/product/${product.id}" class="btn btn-primary flex-grow-1">View Details</a>
                <button class="btn btn-success btn-add-to-cart" data-product-id="${product.id}">
                  <i class="fas fa-cart-plus"></i>
                </button>
              </div>
            </div>
          </div>
        </div>
      `).join('') :
      '<div class="col-12"><div class="no-products"><i class="fas fa-box-open fa-3x mb-3"></i><h3>No products found</h3><p>Try adjusting your search or filters</p></div></div>';

    // Add event listeners to the new buttons
    document.querySelectorAll('.btn-add-to-cart').forEach(button => {
      button.addEventListener('click', addToCart);
    });
  }

  // Add product to cart
  async function addToCart(event) {
    const productId = parseInt(event.target.dataset.productId);
    try {
      const response = await fetch('/api/cart/add', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          product_id: productId,
          quantity: 1
        })
      });

      if (!response.ok) throw new Error('Failed to add to cart');
      await loadCartItems();
      showNotification('Product added to cart!');
      cartSidebar.classList.add('show');
    } catch (error) {
      console.error('Error:', error);
      showNotification('Failed to add product to cart');
    }
  }

  // Load cart items
  async function loadCartItems() {
    let retryCount = 0;
    const maxRetries = 3;
    const baseDelay = 1000;

    while (retryCount < maxRetries) {
      try {
        const response = await fetch('/api/cart/items');
        if (!response.ok) throw new Error(`Server returned ${response.status}: ${response.statusText}`);
        
        const items = await response.json();
        if (!Array.isArray(items)) throw new Error('Invalid cart data format');
        
        updateCartDisplay(items);
        updateCartCount(items);
        return;
      } catch (error) {
        console.error(`Attempt ${retryCount + 1} failed:`, error);
        retryCount++;
        
        if (retryCount === maxRetries) {
          showNotification('Failed to load cart items. Please try again later.');
          cartItems.innerHTML = `
            <div class="alert alert-danger">
              <i class="fas fa-exclamation-circle me-2"></i>
              Unable to load cart items. Please refresh the page.
            </div>`;
        } else {
          const delay = baseDelay * Math.pow(2, retryCount - 1);
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }
  }

  // Update cart display
  function updateCartDisplay(items = cart) {
    cartItems.innerHTML = '';
    let total = 0;
    
    items.forEach(item => {
      const itemTotal = item.price * item.quantity;
      total += itemTotal;
      
      const cartItem = document.createElement('div');
      cartItem.className = 'cart-item';
      cartItem.innerHTML = `
        <div class="d-flex align-items-center">
          <img src="${item.image}" alt="${item.title}" class="cart-item-image">
          <div class="cart-item-details">
            <h6 class="cart-item-title">${item.title}</h6>
            <p class="cart-item-price">$${item.price.toFixed(2)}</p>
            <div class="quantity-controls">
              <button class="quantity-btn" onclick="updateQuantity(${item.id}, ${item.quantity - 1})">-</button>
              <input type="number" class="quantity-input" value="${item.quantity}" min="1" onchange="updateQuantity(${item.id}, this.value)">
              <button class="quantity-btn" onclick="updateQuantity(${item.id}, ${item.quantity + 1})">+</button>
              <button class="remove-item" onclick="removeFromCart(${item.id})">
                <i class="fas fa-trash"></i>
              </button>
            </div>
          </div>
        </div>
      `;
      cartItems.appendChild(cartItem);
    });
    
    cartTotal.textContent = `$${total.toFixed(2)}`;
    checkoutBtn.disabled = items.length === 0;
  }

  // Update cart count
  function updateCartCount(items) {
    const count = items.reduce((total, item) => total + item.quantity, 0);
    cartCount.textContent = count;
  }

  // Show notification
  function showNotification(message) {
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
      notification.remove();
    }, 3000);
  }

  // Event listeners
  searchInput.addEventListener('input', debounceSearch(applyFiltersAndSort, 300));
  filterSelect.addEventListener('change', applyFiltersAndSort);
  sortSelect.addEventListener('change', applyFiltersAndSort);

  cartButton.addEventListener('click', () => {
    cartSidebar.classList.add('show');
    loadCartItems();
  });

  closeCart.addEventListener('click', () => {
    cartSidebar.classList.remove('show');
  });

  // Initialize
  fetchProducts();
  loadCartItems();

  // Make functions available globally
  window.updateQuantity = async function(cartItemId, quantity) {
    if (quantity < 1) return;
    
    try {
      const response = await fetch('/api/cart/update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          cart_item_id: cartItemId,
          quantity: quantity
        })
      });
      
      if (!response.ok) throw new Error('Failed to update quantity');
      await loadCartItems();
    } catch (error) {
      console.error('Error updating quantity:', error);
      showNotification('Failed to update quantity');
    }
  };

  window.removeFromCart = async function(cartItemId) {
    try {
      const response = await fetch('/api/cart/remove', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          cart_item_id: cartItemId
        })
      });
      
      if (!response.ok) throw new Error('Failed to remove item');
      await loadCartItems();
      showNotification('Product removed from cart');
    } catch (error) {
      console.error('Error removing item:', error);
      showNotification('Failed to remove item');
    }
  };
});