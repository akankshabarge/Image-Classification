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
        model = tf.keras.models.load_model('../saved_models/best_model.h5')
        with open('../saved_models/best_model_info.json', 'r') as f:
            model_info = json.load(f)
        return True
    except:
        return False

@app.route('/')
def home():
    model_status = "Ready" if model else "No Model"
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
        <h1>CNN Classifier</h1>
        <div class="status {'ready' if model else 'not-ready'}">{model_status}</div>
        
        <div class="upload">
            <p>Choose an image to classify:</p>
            <input type="file" id="file" accept="image/*">
            <br><br>
            <button class="btn" onclick="classify()">Classify</button>
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
                            '<h3>Result: ' + data.predicted_class + '</h3>' +
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
        return jsonify({'error': 'No model loaded'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        from PIL import Image
        import io
        
        # Read file content properly
        file_content = file.read()
        file.seek(0)  # Reset file pointer
        
        # Open image from bytes
        img = Image.open(io.BytesIO(file_content))
        img = img.convert('RGB')
        img = img.resize((150, 150))
        
        # Convert to array and normalize
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, 0)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Get class name
        classes = ['cats', 'dogs']
        predicted_class = classes[predicted_class_idx]
        
        return jsonify({
            'success': True, 
            'predicted_class': predicted_class, 
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'})



if __name__ == '__main__':
    print("Starting CNN Classifier")
    if load_model():
        print("Model loaded")
    else:
        print("No model found - run train_models.py first")
    
    print("Open: http://localhost:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)    