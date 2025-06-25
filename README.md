
#  CNN-Based Image Classification System

A complete deep learning project that classifies images (e.g., cats vs dogs) using both a **Custom CNN** and **Transfer Learning with ResNet50**. The system evaluates both models and automatically selects the one with better performance. It also includes a **Flask-based web app** for real-time image predictions.

---

##  Project Structure

```
.
├── app.py                  # Flask backend
├── best_model.h5          # Saved best model (CNN or ResNet)
├── train_model.py         # Training script for both models
├── utils.py               # Image preprocessing utilities
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html         # Frontend UI
├── static/
│   └── styles.css         # Optional CSS styling
└── README.md              # Project documentation
```

---

##  Features

* 1. Trains a **Custom CNN** from scratch
* 2. Implements **Transfer Learning** using **ResNet50**
* 3. Automatically selects the better model based on validation accuracy
* 4. Normalizes and augments image data
* 5. Deploys a **Flask web app** for live predictions
* 6. Returns prediction label and confidence score

---

##  Model Architectures

###  Custom CNN

* -> 4 convolutional blocks
* -> Batch Normalization, MaxPooling
* -> GlobalAveragePooling
* -> Dense layers with Dropout
* -> Output layer with Softmax (binary classification)

###  Transfer Learning (ResNet50)

* -> Pre-trained on ImageNet
* -> Frozen base layers
* -> Custom top: GlobalAveragePooling + Dense + Softmax

---

##  Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/image-classifier-cnn.git
   cd image-classifier-cnn
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download or place your image dataset**

   * Structure should be:

     ```
     data/
     ├── train/
     │   ├── cats/
     │   └── dogs/
     ├── val/
     │   ├── cats/
     │   └── dogs/
     ```

5. **Train the models**

   ```bash
   python train_model.py
   ```

6. **Run the Flask app**

   ```bash
   python app.py
   ```

---

## 🌐 Web Interface

The app provides a simple UI to:

* Upload an image
* See model predictions
* View prediction confidence

📍 Navigate to: `http://127.0.0.1:5000/`

---

## 🧪 Sample Prediction Output

```json
{
  "prediction": "Cat",
  "confidence": 0.94
}
```

---

##  Performance Metrics

* **Validation Accuracy** for both models
* **Confidence Score** for each prediction

The model with the **higher validation accuracy** is saved and used for deployment.

---

##  Concepts Used

* Convolutional Neural Networks (CNN)
* Transfer Learning with ResNet50
* Data Augmentation (rotation, flip, zoom)
* Softmax, Dropout, Batch Normalization
* Flask API for deployment

---

##  Technologies

| Tool             | Use                     |
| ---------------- | ----------------------- |
| Python           | Programming language    |
| TensorFlow/Keras | Deep learning framework |
| Flask            | Web app backend         |
| HTML/CSS/JS      | Frontend                |
| PIL              | Image processing        |
| NumPy            | Numerical operations    |

---

##  Customization

* 1. To use more classes: Update the final Dense layer size
* 2. To use different image sizes: Change target size in training & preprocessing
* 3. To deploy online: Use Heroku, AWS, or Render

---

##  Future Improvements

* 1. Add support for multi-class classification
* 2. Use more advanced models (EfficientNet, MobileNet)
* 3. Deploy with Docker or FastAPI
* 4. Add logging and monitoring

---

