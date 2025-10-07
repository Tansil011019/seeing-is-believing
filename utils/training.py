"""
Training and evaluation utilities for deep learning models.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score
from PIL import Image
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_images(images: List[Image.Image], target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess PIL images for model training.
    
    Args:
        images (List[Image.Image]): List of PIL Image objects
        target_size (Tuple[int, int]): Target size for resizing (default: (224, 224))
    
    Returns:
        np.ndarray: Preprocessed image array of shape (n_samples, height, width, channels)
    """
    processed_images = []
    
    for img in images:
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        processed_images.append(img_array)
    
    return np.array(processed_images)


def train_and_evaluate_model(X: List[Image.Image], 
                           y: np.ndarray, 
                           model: keras.Model,
                           test_ratio: float = 0.2,
                           random_seed: int = 42,
                           epochs: int = 50,
                           batch_size: int = 32,
                           validation_split: float = 0.1,
                           target_size: Tuple[int, int] = (224, 224),
                           verbose: int = 1) -> Tuple[keras.Model, np.ndarray, Dict[str, float]]:
    """
    Train and evaluate a TensorFlow model on image classification task.
    
    Args:
        X (List[Image.Image]): List of PIL Image objects
        y (np.ndarray): Labels array
        model (keras.Model): TensorFlow model to train
        test_ratio (float): Ratio of test set (default: 0.2)
        random_seed (int): Random seed for reproducibility (default: 42)
        epochs (int): Number of training epochs (default: 50)
        batch_size (int): Training batch size (default: 32)
        validation_split (float): Validation split ratio (default: 0.1)
        target_size (Tuple[int, int]): Target image size (default: (224, 224))
        verbose (int): Verbosity level (default: 1)
    
    Returns:
        Tuple[keras.Model, np.ndarray, Dict[str, float]]: 
            - Trained model
            - Confusion matrix
            - Dictionary of evaluation metrics
    """
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    # Preprocess images
    if verbose > 0:
        print("Preprocessing images...")
    X_processed = preprocess_images(X, target_size)
    
    # Convert labels to categorical if multi-class
    num_classes = len(np.unique(y))
    if num_classes > 2:
        y_categorical = keras.utils.to_categorical(y, num_classes)
    else:
        y_categorical = y.reshape(-1, 1)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_categorical, 
        test_size=test_ratio, 
        random_state=random_seed,
        stratify=y
    )
    
    if verbose > 0:
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        print(f"Number of classes: {num_classes}")
    
    # Compile model if not already compiled
    if not model.optimizer:
        if num_classes > 2:
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    # Train model
    if verbose > 0:
        print("Starting training...")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose
    )
    
    # Make predictions on test set
    if verbose > 0:
        print("Evaluating model...")
    
    y_pred_proba = model.predict(X_test, verbose=0)
    
    if num_classes > 2:
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
    else:
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_test_labels = y_test.flatten()
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test_labels, y_pred)
    
    # Calculate evaluation metrics
    metrics = {
        'accuracy': accuracy_score(y_test_labels, y_pred),
        'f1_macro': f1_score(y_test_labels, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test_labels, y_pred, average='weighted'),
        'precision_macro': precision_score(y_test_labels, y_pred, average='macro'),
        'precision_weighted': precision_score(y_test_labels, y_pred, average='weighted'),
        'recall_macro': recall_score(y_test_labels, y_pred, average='macro'),
        'recall_weighted': recall_score(y_test_labels, y_pred, average='weighted')
    }
    
    if num_classes == 2:
        metrics.update({
            'f1': f1_score(y_test_labels, y_pred),
            'precision': precision_score(y_test_labels, y_pred),
            'recall': recall_score(y_test_labels, y_pred)
        })
    
    if verbose > 0:
        print("\nEvaluation Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    return model, cm, metrics


def plot_training_history(history: keras.callbacks.History, save_path: str = None) -> None:
    """
    Plot training history including loss and accuracy curves.
    
    Args:
        history (keras.callbacks.History): Training history object
        save_path (str): Path to save the plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training & validation loss
    ax1.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training & validation accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, 
                         class_names: List[str] = None,
                         normalize: bool = False,
                         save_path: str = None) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm (np.ndarray): Confusion matrix
        class_names (List[str]): List of class names (optional)
        normalize (bool): Whether to normalize the matrix (default: False)
        save_path (str): Path to save the plot (optional)
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()