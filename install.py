#!/usr/bin/env python3
"""
🚀 Complete Auto-Installer for Simple CNN Project
Save this file anywhere and run: python install.py
"""
import os
import sys
import subprocess

def create_project_structure():
    """Create the project directory and structure"""
    project_name = "simple_cnn_project"
    
    # Create main project directory
    if os.path.exists(project_name):
        print(f"⚠️  Directory {project_name} already exists!")
        choice = input("Do you want to continue? (y/n): ")
        if choice.lower() != 'y':
            return None
    
    os.makedirs(project_name, exist_ok=True)
    
    # Create subdirectories
    dirs = [
        f"{project_name}/data/train/cats",
        f"{project_name}/data/train/dogs", 
        f"{project_name}/data/test/cats",
        f"{project_name}/data/test/dogs",
        f"{project_name}/models",
        f"{project_name}/uploads"
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return project_name

def create_requirements_txt(project_dir):
    """Create requirements.txt"""
    content = """tensorflow==2.13.0
flask==2.3.2
pillow==10.0.0
numpy==1.24.3
matplotlib==3.7.1
"""
    with open(f"{project_dir}/requirements.txt", 'w') as f:
        f.write(content)
    print("✅ Created: requirements.txt")

def create_setup_py(project_dir):
    """Create setup.py"""
    content = '''#!/usr/bin/env python3
import os

def main():
    print("🛠️  Project Setup Check")
    print("=" * 30)
    
    # Check data folders
    train_cats = len([f for f in os.listdir("data/train/cats") if f.endswith(('.jpg', '.png', '.jpeg'))])
    train_dogs = len([f for f in os.listdir("data/train/dogs") if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    print(f"📁 Training images:")
    print(f"   Cats: {train_cats}")
    print(f"   Dogs: {train_dogs}")
    
    if train_cats > 0 and train_dogs > 0:
        print("✅ Ready for training!")
        print("Run: python train.py")
    else:
        print("⚠️  Add images to data/train/cats/ and data/train/dogs/")
        print("Need at least 10 images per class")

if __name__ == "__main__":
    main()
'''
    with open(f"{project_dir}/setup.py", 'w') as f:
        f.write(content)
    print("✅ Created: setup.py")

def create_train_py(project_dir):
    """Create train.py"""
    content = '''#!/usr/bin/env python3
"""Simple CNN Training Script"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_custom_cnn():
    """Create simple CNN"""
    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(2, activation='softmax')  # 2 classes: cats, dogs
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_transfer_model():
    """Create ResNet transfer learning model"""
    base_model = keras.applications.ResNet50(
        weights='imagenet', include_top=False, input_shape=(150, 150, 3)
    )
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_data():
    """Load and preprocess data"""
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, horizontal_flip=True, validation_split=0.2
    )
    
    train_gen = train_datagen.flow_from_directory(
        'data/train', target_size=(150, 150), batch_size=32,
        class_mode='sparse', subset='training'
    )
    
    val_gen = train_datagen.flow_from_directory(
        'data/train', target_size=(150, 150), batch_size=32,
        class_mode='sparse', subset='validation'
    )
    
    return train_gen, val_gen

def main():
    print("🚀 Training CNN Models")
    print("=" * 30)
    
    # Check data
    if not os.path.exists('data/train/cats') or not os.path.exists('data/train/dogs'):
        print("❌ Please add images to data/train/cats and data/train/dogs")
        return
    
    # Count images
    cats = len([f for f in os.listdir('data/train/cats') if f.endswith(('.jpg', '.png', '.jpeg'))])
    dogs = len([f for f in os.listdir('data/train/dogs') if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if cats < 5 or dogs < 5:
        print(f"❌ Need at least 5 images per class. Found: cats={cats}, dogs={dogs}")
        return
    
    print(f"📊 Found {cats} cat images, {dogs} dog images")
    
    # Load data
    train_gen, val_gen = load_data()
    
    results = {}
    
    # Train Custom CNN
    print("\\n🏋️  Training Custom CNN...")
    custom_model = create_custom_cnn()
    custom_history = custom_model.fit(train_gen, epochs=10, validation_data=val_gen, verbose=1)
    custom_model.save('models/custom_cnn.h5')
    custom_acc = max(custom_history.history['val_accuracy'])
    results['custom_cnn'] = custom_acc
    print(f"Custom CNN Accuracy: {custom_acc:.4f}")
    
    # Train Transfer Learning
    print("\\n🏋️  Training ResNet Transfer Learning...")
    transfer_model = create_transfer_model()
    transfer_history = transfer_model.fit(train_gen, epochs=8, validation_data=val_gen, verbose=1)
    transfer_model.save('models/resnet_transfer.h5')
    transfer_acc = max(transfer_history.history['val_accuracy'])
    results['resnet_transfer'] = transfer_acc
    print(f"ResNet Transfer Accuracy: {transfer_acc:.4f}")
    
    # Save best model
    best_model = 'custom_cnn' if custom_acc > transfer_acc else 'resnet_transfer'
    best_acc = max(custom_acc, transfer_acc)
    
    print(f"\\n🏆 Best Model: {best_model} ({best_acc:.4f})")
    
    # Copy best model
    import shutil
    shutil.copy(f'models/{best_model}.h5', 'models/best_model.h5')
    
    # Save model info
    model_info = {
        'model_type': best_model,
        'accuracy': float(best_acc),
        'class_names': ['cats', 'dogs'],
        'num_classes': 2
    }
    
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Training complete!")
    print("🌐 Run: python app.py")

if __name__ == "__main__":
    main()
'''
    with open(f"{project_dir}/train.py", 'w') as f:
        f.write(content)
    print("✅ Created: train.py")

def create_app_py(project_dir):
    """Create app.py"""
    content = '''#!/usr/bin/env python3
"""Simple Flask Web App"""
import os
import json
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = None
model_info = {}

def load_model():
    global model, model_info
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('models/best_model.h5')
        with open('models/model_info.json', 'r') as f:
            model_info = json.load(f)
        return True
    except:
        return False

@app.route('/')
def home():
    model_status = "✅ Ready" if model else "❌ No Model"
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CNN Classifier</title>
        <style>
            body {{ font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px; }}
            .status {{ padding: 10px; margin: 20px 0; border-radius: 5px; text-align: center; }}
            .ready {{ background: #d4edda; color: #155724; }}
            .not-ready {{ background: #f8d7da; color: #721c24; }}
            .upload {{ border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }}
            .btn {{ background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }}
            .result {{ margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 5px; display: none; }}
            .error {{ background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; display: none; }}
            img {{ max-width: 300px; margin: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <h1>🧠 Simple CNN Classifier</h1>
        <div class="status {'ready' if model else 'not-ready'}">{model_status}</div>
        
        <div class="upload">
            <p>Choose an image to classify:</p>
            <input type="file" id="file" accept="image/*">
            <br><br>
            <button class="btn" onclick="classify()">🔮 Classify</button>
        </div>
        
        <div id="preview"></div>
        <div id="error" class="error"></div>
        <div id="result" class="result"></div>
        
        <script>
            document.getElementById('file').addEventListener('change', function(e) {{
                const file = e.target.files[0];
                if (file) {{
                    const reader = new FileReader();
                    reader.onload = function(e) {{
                        document.getElementById('preview').innerHTML = 
                            '<img src="' + e.target.result + '" alt="Preview">';
                    }};
                    reader.readAsDataURL(file);
                }}
            }});
            
            function classify() {{
                const fileInput = document.getElementById('file');
                const file = fileInput.files[0];
                
                if (!file) {{
                    alert('Please select an image');
                    return;
                }}
                
                const formData = new FormData();
                formData.append('file', file);
                
                fetch('/predict', {{
                    method: 'POST',
                    body: formData
                }})
                .then(response => response.json())
                .then(data => {{
                    if (data.success) {{
                        document.getElementById('result').innerHTML = 
                            '<h3>🏆 Result: ' + data.predicted_class + '</h3>' +
                            '<p>Confidence: ' + (data.confidence * 100).toFixed(1) + '%</p>';
                        document.getElementById('result').style.display = 'block';
                        document.getElementById('error').style.display = 'none';
                    }} else {{
                        document.getElementById('error').textContent = data.error;
                        document.getElementById('error').style.display = 'block';
                        document.getElementById('result').style.display = 'none';
                    }}
                }})
                .catch(error => {{
                    document.getElementById('error').textContent = 'Error: ' + error.message;
                    document.getElementById('error').style.display = 'block';
                }});
            }}
        </script>
    </body>
    </html>
    """

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({{'error': 'No model loaded'}})
    
    if 'file' not in request.files:
        return jsonify({{'error': 'No file uploaded'}})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({{'error': 'No file selected'}})
    
    try:
        from PIL import Image
        
        # Process image
        img = Image.open(file.stream)
        img = img.convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, 0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        class_names = model_info.get('class_names', ['cats', 'dogs'])
        predicted_class = class_names[predicted_class_idx]
        
        return jsonify({{
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence
        }})
        
    except Exception as e:
        return jsonify({{'error': str(e)}})

if __name__ == '__main__':
    print("🚀 Starting CNN Classifier")
    if load_model():
        print("✅ Model loaded")
    else:
        print("⚠️  No model found - run train.py first")
    
    print("🌐 Open: http://localhost:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)
'''
    with open(f"{project_dir}/app.py", 'w') as f:
        f.write(content)
    print("✅ Created: app.py")

def create_readme_files(project_dir):
    """Create README files in data folders"""
    readme_content = """Add your images here!
Supported formats: JPG, PNG, JPEG, GIF
Minimum 10 images recommended for better accuracy.

Example images:
- For cats: cat1.jpg, cat2.png, etc.
- For dogs: dog1.jpg, dog2.png, etc.
"""
    
    folders = ['data/train/cats', 'data/train/dogs', 'data/test/cats', 'data/test/dogs']
    for folder in folders:
        with open(f"{project_dir}/{folder}/README.txt", 'w') as f:
            f.write(readme_content)
    
    print("✅ Created README files in data folders")

def install_dependencies():
    """Install Python packages"""
    print("📦 Installing dependencies...")
    packages = ['tensorflow', 'flask', 'pillow', 'numpy', 'matplotlib']
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
            print(f"✅ {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def main():
    """Main installer function"""
    print("🚀 CNN Project Auto-Installer")
    print("=" * 40)
    
    # Create project structure
    print("\\n1. Creating project structure...")
    project_dir = create_project_structure()
    if not project_dir:
        return
    
    # Create all files
    print("\\n2. Creating project files...")
    create_requirements_txt(project_dir)
    create_setup_py(project_dir)
    create_train_py(project_dir)
    create_app_py(project_dir)
    create_readme_files(project_dir)
    
    # Install dependencies
    print("\\n3. Installing dependencies...")
    install_choice = input("Install Python packages now? (y/n): ")
    if install_choice.lower() == 'y':
        install_dependencies()
    
    print("\\n🎉 Installation Complete!")
    print("=" * 40)
    print(f"📁 Project created in: {project_dir}")
    print("\\n📋 Next steps:")
    print(f"1. cd {project_dir}")
    print("2. Add images to data/train/cats/ and data/train/dogs/")
    print("3. python train.py")
    print("4. python app.py")
    print("5. Open http://localhost:5000")
    
    if install_choice.lower() != 'y':
        print("\\n📦 To install dependencies later:")
        print("pip install tensorflow flask pillow numpy matplotlib")

if __name__ == "__main__":
    main()