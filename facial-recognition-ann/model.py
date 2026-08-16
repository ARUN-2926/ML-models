"""
model.py

Contains the neural network architecture and PCA preprocessing pipeline
for the facial recognition machine learning project.
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def build_pca(n_components: int = 100, whiten: bool = True, random_state: int = 0) -> PCA:
    """
    Constructs and returns a Principal Component Analysis (PCA) model.
    
    Args:
        n_components: Number of principal components to keep.
        whiten: Whether to apply whitening.
        random_state: Seed for reproducibility.
        
    Returns:
        Configured PCA instance.
    """
    return PCA(n_components=n_components, whiten=whiten, random_state=random_state)

def build_knn(n_neighbors: int = 1) -> KNeighborsClassifier:
    """
    Constructs and returns a K-Nearest Neighbors (KNN) classifier.
    
    Args:
        n_neighbors: Number of neighbors to use.
        
    Returns:
        Configured KNN instance.
    """
    return KNeighborsClassifier(n_neighbors=n_neighbors)

def build_ann(input_shape: int, num_classes: int) -> Sequential:
    """
    Builds and compiles the Artificial Neural Network (ANN) model architecture.
    
    Args:
        input_shape: Number of features in the input data.
        num_classes: Number of distinct classes (people) to predict.
        
    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(300, activation='relu'),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
