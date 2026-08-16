import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def main():
    # 1. Importing Python libraries (done above)
    print("--- Iris Flower Classification with Decision Tree ---")
    
    # 2. Loading the Iris dataset
    print("\nLoading the Iris dataset...")
    iris = load_iris()
    
    # Create a DataFrame for easier data manipulation and exploration
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    # Map target integers to species names
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species_name'] = df['species'].map(species_map)

    # Save the dataset to a CSV file for physical inspection
    df.to_csv('iris_dataset.csv', index=False)
    print("Saved the dataset to iris_dataset.csv")

    # 3. Checking the dataset
    print("\n--- Dataset Exploration ---")
    print("head():\n", df.head())
    print("\nisnull().sum():\n", df.isnull().sum())
    print("\ndtypes:\n", df.dtypes)
    print("\ndescribe():\n", df.describe())

    # 4. Pair plotting using Seaborn
    print("\nGenerating Pair Plot...")
    sns.pairplot(df, hue='species_name', diag_kind='hist', markers=["o", "s", "D"])
    plt.suptitle("Pairplot of Iris Dataset", y=1.02)
    plt.savefig('images/pairplot.png')
    plt.close()
    print("Saved pairplot to images/pairplot.png")

    # 5. Splitting the dataset into training and testing data
    X = df.drop(['species', 'species_name'], axis=1)
    y = df['species']
    # Use random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 6. Training a Decision Tree Classifier
    print("\nTraining the Decision Tree Classifier...")
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # 7. Predicting the test data
    print("Making predictions on test data...")
    predictions = clf.predict(X_test)

    # 8. Generating a classification report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, predictions, target_names=iris.target_names))

    # 9. Generating a confusion matrix
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    
    # Visualize and save the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig('images/confusion_matrix.png')
    plt.close()
    print("Saved confusion matrix to images/confusion_matrix.png")

    # 10. Visualizing the Decision Tree
    print("\nGenerating Decision Tree visualization...")
    plt.figure(figsize=(15, 10))
    plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
    plt.title("Decision Tree for Iris Classification")
    plt.savefig('images/decision_tree.png')
    plt.close()
    print("Saved decision tree to images/decision_tree.png")

    # 11. Feeding new flower measurements to the trained classifier
    print("\n--- Predict New Flower ---")
    print("Enter the measurements for a new Iris flower:")
    
    # Fallback to default values for non-interactive execution
    try:
        sepal_length = float(input("Sepal Length (cm) [e.g. 5.1]: ") or 5.1)
        sepal_width = float(input("Sepal Width (cm) [e.g. 3.5]: ") or 3.5)
        petal_length = float(input("Petal Length (cm) [e.g. 1.4]: ") or 1.4)
        petal_width = float(input("Petal Width (cm) [e.g. 0.2]: ") or 0.2)
    except EOFError:
        # Handling the case where the script is run in an environment without standard input
        print("Standard input not available. Using default values for Setosa.")
        sepal_length, sepal_width, petal_length, petal_width = 5.1, 3.5, 1.4, 0.2
    except ValueError:
         print("Invalid input. Using default values for Setosa.")
         sepal_length, sepal_width, petal_length, petal_width = 5.1, 3.5, 1.4, 0.2

    new_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]], 
        columns=iris.feature_names
    )

    # 12. Predicting Setosa, Versicolor, or Virginica
    new_prediction = clf.predict(new_data)
    predicted_class = iris.target_names[new_prediction[0]]
    
    print("\n--- Final Prediction Result ---")
    print(f"Measurements: {sepal_length}, {sepal_width}, {petal_length}, {petal_width}")
    print(f"The predicted class is: {predicted_class.capitalize()}")

if __name__ == "__main__":
    main()
