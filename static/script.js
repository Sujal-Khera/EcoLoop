document.addEventListener('DOMContentLoaded', function() {
    const imageUpload = document.getElementById('imageUpload');
    const uploadBtn = document.getElementById('uploadBtn');
    const preview = document.getElementById('preview');
    const resultContainer = document.getElementById('resultContainer');
    const classificationResult = document.getElementById('classificationResult');
    const wasteType = document.getElementById('wasteType');
    const confidence = document.getElementById('confidence');
    const instructions = document.getElementById('instructions');
    const locationStatus = document.getElementById('locationStatus');
    const recyclingLocations = document.getElementById('recyclingLocations');
    const loadingSpinner = document.getElementById('loadingSpinner');
    
    // Preview image when selected
    imageUpload.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = 'block';
            }
            reader.readAsDataURL(file);
        }
    });
    
    // Handle image upload and classification
    uploadBtn.addEventListener('click', async function() {
        const file = imageUpload.files[0];
        if (!file) {
            alert('Please select an image first.');
            return;
        }
        
        // Show loading spinner
        loadingSpinner.style.display = 'inline-block';
        uploadBtn.disabled = true;
        resultContainer.style.display = 'none';
        
        try {
            // Upload and classify image
            const formData = new FormData();
            formData.append('image', file);
            
            const response = await fetch('/api/classify', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Classification failed');
            }
            
            const result = await response.json();
            
            // Display classification results
            displayClassificationResult(result);
            
            // Get user location and find recycling centers
            getLocationAndFindCenters(result.waste_type);
            
        } catch (error) {
            console.error('Error:', error);
            classificationResult.className = 'alert alert-danger';
            classificationResult.textContent = 'An error occurred during classification. Please try again.';
            resultContainer.style.display = 'block';
        } finally {
            loadingSpinner.style.display = 'none';
            uploadBtn.disabled = false;
        }
    });
    
    // Display classification results
    function displayClassificationResult(result) {
        resultContainer.style.display = 'block';
        
        const wasteTypeInfo = {
            'recyclable': {
                color: 'success',
                instructions: 'This item can be recycled. Please make sure it is clean and dry before recycling.'
            },
            'compostable': {
                color: 'warning',
                instructions: 'This item can be composted. Place it in your compost bin or take it to a composting facility.'
            },
            'general_waste': {
                color: 'danger',
                instructions: 'This item should go in the general waste bin. It cannot be recycled or composted.'
            }
        };
        
        const type = result.waste_type;
        const info = wasteTypeInfo[type] || {color: 'info', instructions: 'Please check local guidelines for disposal.'};
        
        classificationResult.className = `alert alert-${info.color}`;
        classificationResult.textContent = `This item is classified as ${type.replace('_', ' ')}.`;
        
        wasteType.textContent = type.replace('_', ' ');
        confidence.textContent = `${(result.confidence * 100).toFixed(2)}%`;
        instructions.innerHTML = `<p class="mt-3">${info.instructions}</p>`;
    }
    
    // Get user location and find nearby recycling centers
    async function getLocationAndFindCenters(wasteType) {
        locationStatus.innerHTML = '<p>Searching for nearby recycling centers...</p>';
        
        try {
            // Get user's geolocation
            const position = await new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(resolve, reject, {
                    enableHighAccuracy: true,
                    timeout: 5000,
                    maximumAge: 0
                });
            });
            
            const { latitude, longitude } = position.coords;
            
            // Call API to find nearby recycling centers
            const response = await fetch(`/api/recycling-centers?lat=${latitude}&lng=${longitude}&type=${wasteType}`);
            
            if (!response.ok) {
                throw new Error('Could not find recycling centers');
            }
            
            const centers = await response.json();
            displayRecyclingCenters(centers);
            
        } catch (error) {
            console.error('Location error:', error);
            locationStatus.innerHTML = '<p class="text-danger">Could not access your location or find recycling centers.</p>';
        }
    }
    
    // Display nearby recycling centers
    function displayRecyclingCenters(centers) {
        if (centers.length === 0) {
            locationStatus.innerHTML = '<p>No recycling centers found nearby.</p>';
            return;
        }
        
        locationStatus.innerHTML = `<p>Found ${centers.length} recycling centers near you:</p>`;
        
        let locationsHTML = '';
        centers.forEach(center => {
            locationsHTML += `
                <div class="recycling-location">
                    <h5>${center.name}</h5>
                    <p>${center.address}</p>
                    <p><strong>Distance:</strong> ${center.distance.toFixed(1)} km</p>
                    <p><strong>Accepts:</strong> ${center.accepts.join(', ')}</p>
                    <a href="${center.directions_url}" target="_blank" class="btn btn-sm btn-outline-primary">Get Directions</a>
                </div>
            `;
        });
        
        recyclingLocations.innerHTML = locationsHTML;
    }
});