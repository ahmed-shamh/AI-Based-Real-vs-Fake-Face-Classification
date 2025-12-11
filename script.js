// Theme Management
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeToggle(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeToggle(newTheme);
}

function updateThemeToggle(theme) {
    const toggleBtn = document.querySelector('.theme-toggle');
    if (toggleBtn) {
        const icon = toggleBtn.querySelector('i');
        if (theme === 'dark') {
            icon.className = 'bx bx-sun';
            toggleBtn.title = 'Switch to light mode';
        } else {
            icon.className = 'bx bx-moon';
            toggleBtn.title = 'Switch to dark mode';
        }
    }
}

// Mobile Menu
function toggleMobileMenu() {
    const navLinks = document.querySelector('.nav-links');
    if (navLinks) {
        navLinks.classList.toggle('active');
    }
}

// File Upload Functionality
function initializeFileUpload() {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('fileInput');
    const previewImg = document.getElementById('preview-img');
    const previewContainer = document.getElementById('previewContainer');
    const xceptionBtn = document.getElementById('xceptionBtn');
    const swinBtn = document.getElementById('swinBtn');
    const spinner = document.getElementById('spinner');
    const result = document.getElementById('result');

    console.log('Initializing file upload...');

    if (!dropArea || !fileInput) {
        console.log('File upload elements not found');
        return;
    }

    // Click to select file
    dropArea.addEventListener('click', function(e) {
        fileInput.click();
    });

    // File selection handler
    fileInput.addEventListener('change', function(e) {
        if (this.files && this.files[0]) {
            handleFile(this.files[0]);
        }
    });

    // Drag and drop handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
    });

    // Handle dropped files
    dropArea.addEventListener('drop', function(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    function handleFile(file) {
        console.log('Handling file:', file.name, file.type);
        
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (JPG, PNG)');
            return;
        }
        
        if (previewContainer) {
            previewContainer.style.display = 'block';
        }
        
        if (result) {
            result.className = 'result-container result-pending';
            result.querySelector('.result-icon i').className = 'bx bx-time';
            result.querySelector('.result-content h3').textContent = 'Ready to Analyze';
            result.querySelector('.result-content p').textContent = `Image "${file.name}" loaded. Click a button to analyze.`;
            result.style.display = 'flex';
        }
        
        const reader = new FileReader();
        reader.onload = function(e) {
            if (previewImg) {
                previewImg.src = e.target.result;
                previewImg.style.display = 'block';
            }
        };
        reader.onerror = function(error) {
            console.error('Error reading file:', error);
            alert('Error reading the image file. Please try again.');
        };
        reader.readAsDataURL(file);
    }

    // Xception Button Click
    if (xceptionBtn) {
        xceptionBtn.addEventListener('click', async function() {
            await analyzeImage('xception', xceptionBtn);
        });
    }

    // Swin Button Click
    if (swinBtn) {
        swinBtn.addEventListener('click', async function() {
            await analyzeImage('swin', swinBtn);
        });
    }

    async function analyzeImage(model, button) {
        console.log('Analyze button clicked with model:', model);
        
        if (!fileInput.files[0]) {
            alert('Please select an image first!');
            return;
        }
        
        const file = fileInput.files[0];
        const modelNames = {
            'xception': 'Xception Model',
            'swin': 'Swin Transformer Model'
        };
        
        // Disable both buttons
        xceptionBtn.disabled = true;
        swinBtn.disabled = true;
        
        // Update clicked button
        const originalHTML = button.innerHTML;
        button.classList.add('loading');
        button.innerHTML = '<i class="bx bx-loader-circle bx-spin"></i><span class="btn-text"><span class="btn-title">Analyzing...</span></span>';
        
        if (spinner) {
            spinner.style.display = 'block';
        }
        
        if (result) {
            result.className = 'result-container result-pending';
            result.querySelector('.result-icon i').className = 'bx bx-loader-circle bx-spin';
            result.querySelector('.result-content h3').textContent = 'Analyzing Image';
            result.querySelector('.result-content p').textContent = `Processing with ${modelNames[model]}...`;
            result.style.display = 'flex';
        }
        
        try {
            console.log('Calling API with model:', model);
            const prediction = await analyzeWithAPI(file, model);
            console.log('API response:', prediction);
            
            if (prediction.error) {
                throw new Error(prediction.error);
            }
            
            if (prediction && prediction.prediction !== undefined) {
                displayResults(prediction, result);
            } else {
                throw new Error('Invalid response from API');
            }
            
        } catch (error) {
            console.error('API Error:', error);
            if (result) {
                result.className = 'result-container result-error';
                result.querySelector('.result-icon i').className = 'bx bx-error';
                result.querySelector('.result-content h3').textContent = 'Analysis Failed';
                result.querySelector('.result-content p').textContent = `Error: ${error.message}`;
                result.style.display = 'flex';
            }
        } finally {
            if (spinner) {
                spinner.style.display = 'none';
            }
            // Re-enable both buttons
            xceptionBtn.disabled = false;
            swinBtn.disabled = false;
            button.classList.remove('loading');
            button.innerHTML = originalHTML;
        }
    }
}

// Display results
function displayResults(prediction, result) {
    const isReal = prediction.prediction === 0;
    const modelName = prediction.model || 'Unknown';
    
    if (result) {
        if (isReal) {
            result.className = 'result-container result-real';
            result.querySelector('.result-icon i').className = 'bx bx-check-shield';
            result.querySelector('.result-content h3').textContent = 'Authentic Image ✓';
            result.querySelector('.result-content p').innerHTML = `
                <strong>Classification: REAL</strong><br>
                Model: ${modelName}<br>
                Confidence: <strong>${prediction.confidence}</strong><br>
                Probability: ${(prediction.probability * 100).toFixed(2)}%
            `;
        } else {
            result.className = 'result-container result-fake';
            result.querySelector('.result-icon i').className = 'bx bx-error-alt';
            result.querySelector('.result-content h3').textContent = 'Deep Fake Detected! ⚠️';
            result.querySelector('.result-content p').innerHTML = `
                <strong>Classification: FAKE</strong><br>
                Model: ${modelName}<br>
                Confidence: <strong>${prediction.confidence}</strong><br>
                Probability: ${(prediction.probability * 100).toFixed(2)}%
            `;
        }
    }
}

// API CALL
async function analyzeWithAPI(file, model) {
    console.log('Sending file to API:', file.name, 'with model:', model);
    
    const formData = new FormData();
    formData.append("file", file);

    const API_URL = "http://127.0.0.1:8000";
    
    const endpoints = {
        'xception': `${API_URL}/predict/xception`,
        'swin': `${API_URL}/predict/swin`
    };
    
    const endpoint = endpoints[model];
    
    try {
        console.log(`Calling endpoint: ${endpoint}`);
        const response = await fetch(endpoint, {
            method: "POST",
            body: formData
        });
        
        console.log('API response status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('API error:', errorText);
            throw new Error(`API error: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('API result:', result);
        return result;
        
    } catch (error) {
        console.error('Fetch error:', error);
        return {
            error: `Failed to connect to API: ${error.message}. Make sure backend is running on http://127.0.0.1:8000`,
            prediction: null,
            probability: null,
            label: "Connection Error",
            confidence: "0%"
        };
    }
}

// Initialize everything
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded - initializing...');
    
    initializeTheme();
    initializeFileUpload();
    
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    document.querySelectorAll('.nav-links a').forEach(link => {
        if (link.getAttribute('href') === currentPage) {
            link.classList.add('active');
        }
    });
    
    document.addEventListener('click', function(event) {
        const navLinks = document.querySelector('.nav-links');
        const mobileMenuBtn = document.querySelector('.mobile-menu');
        
        if (navLinks && navLinks.classList.contains('active') && 
            !navLinks.contains(event.target) && 
            !mobileMenuBtn.contains(event.target)) {
            navLinks.classList.remove('active');
        }
    });
    
    console.log('Initialization complete');
});

window.toggleTheme = toggleTheme;
window.toggleMobileMenu = toggleMobileMenu;

document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const navLinks = document.querySelector('.nav-links');
        if (navLinks && navLinks.classList.contains('active')) {
            navLinks.classList.remove('active');
        }
    }
});