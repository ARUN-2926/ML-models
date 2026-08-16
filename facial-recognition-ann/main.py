"""
main.py

Main execution script for the Facial Recognition ML pipeline.
It handles data loading, preprocessing, visualization, and model training.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import mglearn
import os
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from model import build_pca, build_knn, build_ann

# 1. Set Random seeds for reproducibility
np.random.seed(180)
tf.random.set_seed(180)

def main():
    os.makedirs("images", exist_ok=True)
    # 2. Import the data
    print("Loading Labeled Faces in the Wild (LFW) dataset...")
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    print("Data shape:", people.data.shape)
    print("Target shape:", people.target.shape)
    image_shape = people.images[0].shape

    # 3. Display sample faces
    print("Displaying sample faces...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
    for target, image, ax in zip(people.target, people.images, axes.ravel()):
        ax.imshow(image, cmap='gray')
        ax.set_title(people.target_names[target])
    plt.tight_layout()
    plt.savefig("images/sample_faces.png", bbox_inches='tight')
    plt.close()

    # 4. Split the raw data (Exploratory Split)
    X_train, X_test, y_train, y_test = train_test_split(
        people.data, people.target, test_size=0.2, random_state=20
    )
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    print("\nTrain shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)
    print("Number of classes:", len(people.target_names))

    # 5. Count images per person
    print("\nImage counts per person:")
    counts = np.bincount(people.target)
    for i, (count, name) in enumerate(zip(counts, people.target_names)):
        print("{0:25} {1:3}".format(name, count), end=" ")
        if (i + 1) % 3 == 0:
            print()
    print("\n")

    # 6. Transform the data (Subset max 50 images per class for balance)
    mask = np.zeros(people.target.shape, dtype=bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = True
        
    X_people = people.data[mask]
    y_people = people.target[mask]
    
    X_people = X_people / 255.0
    
    # 7. PCA + KNN Pipeline
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
        X_people, y_people, stratify=y_people, random_state=0
    )

    # KNN without PCA
    knn = build_knn(n_neighbors=1)
    knn.fit(X_train_sub, y_train_sub)
    print("Test set score of 1-NN without PCA: {:.2f}".format(knn.score(X_test_sub, y_test_sub)))

    # PCA Whitening plot
    try:
        print("Plotting PCA whitening...")
        mglearn.plots.plot_pca_whitening()
        plt.savefig("images/pca_whitening.png", bbox_inches='tight')
        plt.close()
    except Exception as e:
        print("Could not plot PCA whitening:", e)

    # PCA transformation
    pca = build_pca(n_components=100, whiten=True, random_state=0)
    pca.fit(X_train_sub)
    X_train_pca = pca.transform(X_train_sub)
    X_test_pca = pca.transform(X_test_sub)
    print("X_train_pca shape:", X_train_pca.shape)

    # KNN with PCA
    knn_pca = build_knn(n_neighbors=1)
    knn_pca.fit(X_train_pca, y_train_sub)
    print("Test set accuracy of 1-NN with PCA: {:.2f}".format(knn_pca.score(X_test_pca, y_test_sub)))
    print("PCA components shape:", pca.components_.shape)

    # PCA Components visualization (Eigenfaces)
    print("Displaying PCA components...")
    fig, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
    for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
        ax.imshow(component.reshape(image_shape), cmap='viridis')
        ax.set_title("{}. component".format(i + 1))
    plt.tight_layout()
    plt.savefig("images/eigenfaces.png", bbox_inches='tight')
    plt.close()

    # 8. ANN Model Training
    # MinMax scaling on the PCA-transformed features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)

    # Build ANN model
    num_classes = len(np.unique(y_train_sub))
    ann_model = build_ann(input_shape=100, num_classes=num_classes)

    # Train model
    print("\nTraining Artificial Neural Network (ANN)...")
    history = ann_model.fit(
        X_train_scaled, y_train_sub, 
        batch_size=50, epochs=20, 
        shuffle=False, verbose=1
    )

    # Evaluate model
    print("\nEvaluating ANN Model...")
    loss, accuracy = ann_model.evaluate(X_test_scaled, y_test_sub, verbose=1)
    print("Test Loss: {:.4f}".format(loss))
    print("Test Accuracy: {:.4f}".format(accuracy))

if __name__ == "__main__":
    main()
