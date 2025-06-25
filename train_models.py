import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_custom_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_transfer_model():
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet', include_top=False, input_shape=(150, 150, 3)
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("CNN Training Started")
    
    # Check data
    if not os.path.exists('data/train/cats') or not os.path.exists('data/train/dogs'):
        print("Error: Missing data folders")
        return
    
    # Count images
    cats = len([f for f in os.listdir('data/train/cats') if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    dogs = len([f for f in os.listdir('data/train/dogs') if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    print(f"Found {cats} cats, {dogs} dogs")
    
    if cats < 10 or dogs < 10:
        print("Need at least 10 images per class")
        return
    
    # Create data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    # Train Custom CNN
    print("Training Custom CNN...")
    custom_model = create_custom_cnn()
    custom_history = custom_model.fit(train_generator, epochs=5, validation_data=val_generator)
    custom_acc = max(custom_history.history['val_accuracy'])
    
    # Train Transfer Learning
    print("Training Transfer Learning...")
    transfer_model = create_transfer_model()
    transfer_history = transfer_model.fit(train_generator, epochs=5, validation_data=val_generator)
    transfer_acc = max(transfer_history.history['val_accuracy'])
    
    # Save best model
    if custom_acc > transfer_acc:
        best_model = custom_model
        best_type = 'custom_cnn'
        best_acc = custom_acc
    else:
        best_model = transfer_model
        best_type = 'transfer_learning'
        best_acc = transfer_acc
    
    print(f"Best model: {best_type} with accuracy: {best_acc:.4f}")
    
    # Save model
    os.makedirs('saved_models', exist_ok=True)
    best_model.save('saved_models/best_model.h5')
    
    # Save model info
    model_info = {
        'model_type': best_type,
        'accuracy': float(best_acc),
        'class_indices': {'cats': 0, 'dogs': 1},
        'num_classes': 2
    }
    
    with open('saved_models/best_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("Training complete! Run: cd ui && python app.py")

if __name__ == "__main__":
    main()