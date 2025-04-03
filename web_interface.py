from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import sys
from pathlib import Path
import time
import shutil

# Initialize Flask app
app = Flask(__name__)
current_dir = Path(__file__).resolve().parent

# Configuration
app.config.update(
    UPLOAD_FOLDER=str(current_dir / 'uploads'),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    SAMPLE_FOLDER=str(current_dir / 'ACGPN_inference' / 'sample')
)

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SAMPLE_FOLDER'], exist_ok=True)

# Add ACGPN_inference to Python path
sys.path.append(str(current_dir / 'ACGPN_inference'))

# Check for required dependencies
DEPENDENCIES = {
    'torch': None,
    'PIL': None,
    'cv2': None,
    'numpy': None,
    'base64': None
}

try:
    import torch
    DEPENDENCIES['torch'] = True
except ImportError:
    print("PyTorch is not installed. Please install it using: pip install torch torchvision")

try:
    from PIL import Image
    DEPENDENCIES['PIL'] = True
except ImportError:
    print("Pillow is not installed. Please install it using: pip install pillow")

try:
    import cv2
    DEPENDENCIES['cv2'] = True
except ImportError:
    print("OpenCV is not installed. Please install it using: pip install opencv-python")

try:
    import numpy as np
    DEPENDENCIES['numpy'] = True
except ImportError:
    print("NumPy is not installed. Please install it using: pip install numpy")

try:
    import base64
    DEPENDENCIES['base64'] = True
except ImportError:
    print("Base64 module not found in Python standard library")

# Try importing the try-on module
try:
    from test import generate_try_on
except ImportError as e:
    print(f"Warning: Could not import generate_try_on: {e}")
    print("Please ensure the ACGPN_inference module is properly set up")
    generate_try_on = None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def check_dependencies():
    """Check if all required dependencies are available"""
    missing = [dep for dep, installed in DEPENDENCIES.items() if not installed]
    if missing:
        return False, f"Missing dependencies: {', '.join(missing)}"
    return True, "All dependencies are installed"

def check_model_files():
    """Check if model checkpoint files exist"""
    checkpoint_dir = current_dir / 'ACGPN_inference' / 'checkpoints'
    if not checkpoint_dir.exists():
        return False, "Model checkpoint directory not found"
    
    # Check for specific model files (update this list based on required files)
    required_files = ['latest_net_G.pth']  # Add other required files
    missing_files = [f for f in required_files if not (checkpoint_dir / f).exists()]
    
    if missing_files:
        return False, f"Missing model files: {', '.join(missing_files)}"
    return True, "All model files found"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_files(*files):
    """Safely remove multiple files"""
    for file_path in files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not remove file {file_path}: {e}")

def cleanup_old_files(directory, max_age_hours=24):
    """Clean up files older than max_age_hours"""
    try:
        current_time = time.time()
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.getmtime(filepath) < current_time - (max_age_hours * 3600):
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Warning: Could not remove old file {filepath}: {e}")
    except Exception as e:
        print(f"Warning: Could not clean up old files: {e}")

@app.route('/')
def index():
    # Check system status
    dep_status, dep_msg = check_dependencies()
    model_status, model_msg = check_model_files()
    
    status = {
        'dependencies': dep_msg,
        'model_files': model_msg,
        'system_ready': dep_status and model_status
    }
    
    return render_template('index.html', status=status)

@app.route('/try-on', methods=['POST'])
def try_on():
    # Check dependencies and model files
    dep_status, dep_msg = check_dependencies()
    if not dep_status:
        return jsonify({'error': dep_msg}), 500
    
    model_status, model_msg = check_model_files()
    if not model_status:
        return jsonify({'error': model_msg}), 500
    
    if generate_try_on is None:
        return jsonify({'error': 'Try-on module not properly initialized'}), 500

    if 'person_image' not in request.files or 'clothing_image' not in request.files:
        return jsonify({'error': 'Missing required images'}), 400
    
    person_file = request.files['person_image']
    clothing_file = request.files['clothing_image']
    
    if person_file.filename == '' or clothing_file.filename == '':
        return jsonify({'error': 'No selected files'}), 400
    
    if not (allowed_file(person_file.filename) and allowed_file(clothing_file.filename)):
        return jsonify({'error': 'Invalid file type'}), 400
    
    person_path = None
    clothing_path = None
    result_path = None
    
    try:
        # Clean up old files
        cleanup_old_files(app.config['UPLOAD_FOLDER'])
        cleanup_old_files(app.config['SAMPLE_FOLDER'])
        
        # Save uploaded files with unique names
        timestamp = str(int(time.time()))
        person_path = os.path.join(app.config['UPLOAD_FOLDER'], f'person_{timestamp}_{secure_filename(person_file.filename)}')
        clothing_path = os.path.join(app.config['UPLOAD_FOLDER'], f'clothing_{timestamp}_{secure_filename(clothing_file.filename)}')
        
        person_file.save(person_path)
        clothing_file.save(clothing_path)
        
        # Ensure images are in the correct format and size
        for img_path in [person_path, clothing_path]:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large (max 1024px)
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)
            
            img.save(img_path, quality=95)
        
        # Generate try-on result
        result_path = generate_try_on(person_path, clothing_path)
        
        if not result_path or not os.path.exists(result_path):
            return jsonify({'error': 'Failed to generate try-on result'}), 500
        
        # Read and encode the result image
        result_img = cv2.imread(result_path)
        if result_img is None:
            return jsonify({'error': 'Failed to read generated result'}), 500
            
        # Convert BGR to RGB
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', result_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'result': f'data:image/jpeg;base64,{result_base64}'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing images: {str(e)}'}), 500
        
    finally:
        # Clean up temporary files
        cleanup_files(person_path, clothing_path, result_path)

@app.route('/download/<path:filename>')
def download_file(filename):
    try:
        return send_file(
            os.path.join(app.config['SAMPLE_FOLDER'], filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': f'Error downloading file: {str(e)}'}), 500

@app.route('/status')
def status():
    """Get system status"""
    dep_status, dep_msg = check_dependencies()
    model_status, model_msg = check_model_files()
    
    return jsonify({
        'dependencies': dep_msg,
        'model_files': model_msg,
        'system_ready': dep_status and model_status
    })

if __name__ == '__main__':
    # Print system status on startup
    print("\nSystem Status:")
    dep_status, dep_msg = check_dependencies()
    print(f"Dependencies: {dep_msg}")
    model_status, model_msg = check_model_files()
    print(f"Model files: {model_msg}")
    print(f"System ready: {dep_status and model_status}\n")
    
    app.run(debug=True) 