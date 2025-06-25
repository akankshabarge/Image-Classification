import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, test_generator):
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    
    report = classification_report(true_classes, predicted_classes, target_names=class_labels, output_dict=True)
    cm = confusion_matrix(true_classes, predicted_classes)
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_labels': class_labels
    }

def plot_training_history(history, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title(f'{title} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title(f'{title} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()