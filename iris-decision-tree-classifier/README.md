# Iris Decision Tree Classifier

This project is a complete, beginner-friendly Machine Learning workflow for classifying Iris flowers using a Decision Tree model.

## About the Dataset

This project uses the famous **Iris flower dataset**, which was introduced by the British statistician and biologist Ronald Fisher in his 1936 paper. It is considered a classic in the field of Machine Learning and is often referred to as the "Hello World" of classification problems.

### Dataset Features (Input)
The dataset contains **150 samples** of Iris flowers, evenly distributed across three species. For each flower, the following four measurements (in centimeters) were recorded:
- **Sepal Length**
- **Sepal Width**
- **Petal Length**
- **Petal Width**

### Target Classes (Output)
The goal of the dataset is to classify the samples into one of three species (classes): 
1. **Setosa**
2. **Versicolor**
3. **Virginica**

*Note: In this project, the dataset is loaded directly from the `scikit-learn` library's built-in datasets (`load_iris()`). However, running the scripts will automatically generate and save a physical copy of the data as `iris_dataset.csv` in your project folder for you to explore.*

## Machine Learning Pipeline Explanation

This project follows a standard 12-step Machine Learning pipeline:

1. **Importing Libraries:** We use `pandas` for data manipulation, `numpy` for numerical operations, `matplotlib` & `seaborn` for visualization, and `scikit-learn` for the ML algorithms.
2. **Loading Dataset:** We load the built-in Iris dataset from `scikit-learn`.
3. **Exploring Dataset:** We check the first few rows, missing values, data types, and statistical summaries to understand the data.
4. **Data Visualization:** We create a pair plot to visualize relationships between different measurements.
5. **Data Preparation:** We split the dataset into features (`X` - the measurements) and target (`y` - the species). We then split it into training (70%) and testing (30%) sets.
6. **Model Creation:** We instantiate a Decision Tree Classifier.
7. **Model Training:** We teach the model using the training data (`fit()`).
8. **Making Predictions:** We ask the model to predict the species for the testing data.
9. **Classification Report:** We evaluate how well the model did using metrics like Precision, Recall, and F1-Score.
10. **Confusion Matrix:** We visualize where the model made correct and incorrect predictions.
11. **Visualizing the Tree:** We plot the actual Decision Tree logic to see how it makes decisions.
12. **Predicting New Data:** We provide completely new measurements to the trained model and get a final prediction.

## How the Final Prediction Works

The final section of the script prompts the user (or uses defaults) for four numerical values representing the petal and sepal measurements. The trained Decision Tree passes these numbers through its learned rules (e.g., "Is petal width <= 0.8 cm?") and eventually arrives at a leaf node which outputs the predicted class (Setosa, Versicolor, or Virginica).

## Project Structure

```
iris-decision-tree-classifier/
│
├── iris_classifier.py      # Main Python script
├── iris_classifier.ipynb   # Interactive Jupyter Notebook version
├── requirements.txt        # Required Python libraries
├── README.md               # Project documentation
├── .gitignore              # Ignored files for Git
│
├── images/                 # Generated visualizations
│   ├── pairplot.png
│   ├── confusion_matrix.png
│   └── decision_tree.png
```

## Setup and Execution

### Local Setup

1. **Clone or Download** the project to your local machine.
2. **Open a terminal** and navigate to the project directory:
   ```bash
   cd iris-decision-tree-classifier
   ```
3. **Install the requirements**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the script**:
   ```bash
   python iris_classifier.py
   ```
   *Note: When running, the script will prompt you for new flower measurements. If you press Enter, it will use default values.*

### Generating the Visualizations
When you run `iris_classifier.py`, it automatically generates and saves three image files inside the `images/` folder.

## Creating and Pushing to a GitHub Repository

To share your project on GitHub:

1. Create a new repository on [GitHub](https://github.com/new). Do not initialize it with a README or .gitignore (since you already have them).
2. Open your terminal in the project directory.
3. Initialize the local directory as a Git repository:
   ```bash
   git init
   ```
4. Add all files to the staging area:
   ```bash
   git add .
   ```
5. Commit the files:
   ```bash
   git commit -m "Initial commit: Complete Iris ML Pipeline"
   ```
6. Link your local repository to the GitHub repository (replace `URL` with your repository link):
   ```bash
   git remote add origin <URL>
   ```
7. Push the code to GitHub:
   ```bash
   git branch -M main
   git push -u origin main
   ```

## Future Enhancements: Converting to a Streamlit App

While this basic version uses the command line, it can easily be converted into an interactive web application using **Streamlit**.

To do this in the future:
1. Install Streamlit: `pip install streamlit`
2. Create a new file (e.g., `app.py`).
3. Import streamlit (`import streamlit as st`) instead of using standard `print()` and `input()`.
4. Use `st.sidebar.number_input()` to capture the 4 flower measurements.
5. Use `st.pyplot()` to display the Confusion Matrix and Decision Tree directly on the web page.
6. Display the final prediction using `st.success()`.
