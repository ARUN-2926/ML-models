# Spam Mail Detection Using Support Vector Machine

## 1. Project Overview
This project is a complete end-to-end Machine Learning pipeline that detects whether an email or SMS message is "Spam" or "Not Spam" (Ham). It leverages Natural Language Processing (NLP) techniques and a Support Vector Machine (SVM) to classify text data with high 
Fixing Facial Recognition ANN
if anything is not install . download it as per procedure 
I have fully implemented the Spam Mail Detection Using Support Vector Machine project. Everything has been tested from end to end.

Since you approved automatic downloading, you do not need to manually add any dataset! The pipeline handles fetching the UCI SMS Spam dataset from the internet and formats it perfectly for the experiment.

Please check the Walkthrough artifact to read the full summary of what was accomplished, or dive right into the newly created e:\f1\project for ML\spam-mail-svm folder!

1. Final Project Structure
text
spam-mail-svm/
├── main.py                # Runs the full pipeline end-to-end
├── train.py               # Handles train/test splitting, SVM training, and evaluation
├── predict.py             # CLI application to test the saved model interactively
├── preprocessing.py       # Handles dataset downloading, cleaning, and exploration
├── requirements.txt       # Project dependencies
├── README.md              # Detailed documentation mapping exactly to the 18 requirements
│
├── data/                  
│   └── spam.csv           # Automatically downloaded & cleaned dataset
│
├── models/                
│   ├── spam_classifier.pkl 
│   └── vectorizer.pkl      
│
└── outputs/               
    ├── class_distribution.png              
    ├── confusion_matrix.png                
    ├── confusion_matrix_improved_svm.png   
    └── model_comparison.png
2. Commands to Run It
Navigate to the directory first:

powershell
cd "e:\f1\project for ML\spam-mail-svm"
To run the entire training pipeline and regenerate all models/graphs:

powershell
python main.py
To test a custom message interactively using the saved model:

powershell
python predict.py
3. Summary of the ML Pipeline
Auto-Download: Fetched SMS Spam Collection and saved it as spam.csv.
Cleaning: Dropped 403 duplicate entries.
Splitting: 80/20 train/test split utilizing stratification due to class imbalance.
Vectorization: CountVectorizer to convert text into numerical feature matrices (yielding 7,414 features).
Training Baseline: Trained a default SVM (RBF Kernel).
Training Improved Model: Ran GridSearchCV testing linear and RBF kernels with varying C margins.
Saving Artifacts: Visualizations mapped to outputs/ and final artifacts .pkl saved to models/.
4. Actual Model Performance
Baseline SVM (RBF Kernel, C=1.0):

Accuracy: 97.49%
Precision: 100.00%
Recall: 80.15% (Caught 80% of actual spam)
F1-Score: 88.98%
Improved SVM (Linear Kernel, C=0.1):

Accuracy: 97.68%
Precision: 99.08%
Recall: 82.44% (Caught over 82% of actual spam)
F1-Score: 90.00%
Note: The Linear Kernel provided a solid improvement in detecting actual spam (Recall) which raised the overall F1-Score.

5. Summary of README Documentation
I have generated an extensive, 18-section README.md exactly as requested. It documents:

Detailed instructions on what each script does and how data flows.
The exact hyperparameters identified by GridSearchCV.
Step-by-step breakdown of text vectorization (CountVectorizer) in plain terms.
Precise instructions for environment setup, model training, and testing new accuracy. 

## 2. Problem Statement
With the increasing volume of digital communication, users are frequently targeted by unsolicited, irrelevant, or malicious messages known as "spam". Identifying and filtering these messages automatically is crucial for maintaining productivity and security.

## 3. Aim
To implement a Spam Mail Detection system using a Support Vector Machine (SVM) as per Experiment 2 of the Machine Learning Lab Manual.

## 4. Objectives
* Build a reproducible Machine Learning pipeline.
* Automatically fetch and prepare a text-based dataset.
* Transform textual data into numerical features using `CountVectorizer`.
* Train a baseline Support Vector Machine classifier.
* Improve the model using Hyperparameter tuning (`GridSearchCV`).
* Evaluate the model thoroughly using multiple metrics (Precision, Recall, F1-Score, Confusion Matrix).
* Save the best-performing model for future inference.

## 5. Dataset
* **Source:** The standard UCI "SMS Spam Collection" dataset. (Automatically downloaded if not present).
* **Columns:** `Category` (spam/ham) and `Message` (text of the message).
* **Size:** Initially 5,572 records. After dropping 403 duplicates, the dataset contains **5,169 distinct records**.
* **Class Distribution:** 4,516 non-spam messages (~87.4%) and 653 spam messages (~12.6%).

## 6. Technologies Used
* **Python**: Core programming language.
* **Pandas**: Used for data manipulation, cleaning, and DataFrame management.
* **NumPy**: Used for numerical operations and array processing.
* **Scikit-learn**: Used for Data Splitting (`train_test_split`), NLP Vectorization (`CountVectorizer`), Classification (`SVC`), Hyperparameter Tuning (`GridSearchCV`), and Metrics.
* **Matplotlib & Seaborn**: Used for generating graphs and the confusion matrix.
* **Joblib**: Used for saving and loading the trained model and vectorizer.
* **Requests & Urllib**: Used for fetching the dataset dynamically from the web.

## 7. Project Structure
```text
spam-mail-svm/
│
├── main.py                # Main orchestrator script to run the entire ML pipeline
├── train.py               # Handles train/test splitting, SVM training, and evaluation
├── predict.py             # CLI application for users to test the saved model
├── preprocessing.py       # Handles dataset downloading, cleaning, and exploration
├── requirements.txt       # Contains Python dependencies required for the project
├── README.md              # Detailed project documentation
│
├── data/                  
│   └── spam.csv           # The final cleaned dataset used for training
│
├── models/                
│   ├── spam_classifier.pkl # Saved Best SVM model
│   └── vectorizer.pkl      # Saved CountVectorizer
│
└── outputs/               
    ├── class_distribution.png              # Bar chart of Spam vs Ham
    ├── confusion_matrix.png                # Confusion matrix of baseline SVM
    ├── confusion_matrix_improved_svm.png   # Confusion matrix of the best model
    └── model_comparison.png                # Visual comparison of both models
```

## 8. Implementation Steps

### Step 1 — Environment Setup
Dependencies were documented in `requirements.txt`. They can be installed into a Python virtual environment to keep global packages clean.

### Step 2 — Dataset Preparation
In `preprocessing.py`, the script checks if `data/spam.csv` exists. If not, it automatically downloads the UCI SMS Spam zip file, unzips it, formats it into a CSV file with "Category" and "Message" headers, and saves it locally.

### Step 3 — Loading the Dataset
Pandas was used (`pd.read_csv`) to load the dataset for cleaning and exploration.

### Step 4 — Dataset Exploration
Exploration showed the dataset contains 5,572 rows and 2 columns. There were no missing values (Nulls), but there were duplicates. The `Category` column distribution was analyzed to understand class imbalance.

### Step 5 — Data Cleaning
403 duplicate messages were identified and removed using `df.drop_duplicates(keep='first')` to prevent data leakage and overfitting. The target variable `Category` was mapped to binary values (`spam`: 1, `ham`: 0).

### Step 6 — Train/Test Split
The dataset was split using an 80/20 ratio. `stratify=y` was strictly used during the split to ensure both the training and test sets contained the exact same proportion of Spam vs Ham messages, which is vital due to class imbalance.

### Step 7 — Text Vectorization
`CountVectorizer` was applied. It converts the text into numerical features by building a vocabulary of all words in the dataset and representing each message as a vector of word counts. `stop_words='english'` was used to remove common meaningless words (like "the", "and", "is"). The vectorizer was fitted *only* on the training data. This resulted in a feature space of 7,414 distinct words.

### Step 8 — SVM Model
Support Vector Machine (SVM) was chosen because it works exceptionally well in high-dimensional spaces (like our 7,414 features). A baseline SVM using the default Radial Basis Function (RBF) kernel was established.

### Step 9 — Model Training
The model was trained (`model.fit`) on the numerical matrices returned by the CountVectorizer. To improve the baseline, we applied `GridSearchCV` to automatically test multiple hyperparameters (`C=[0.1, 1, 10]`, `kernel=['linear', 'rbf']`). The best configuration was found to be a **Linear Kernel with C=0.1**.

### Step 10 — Prediction
Predictions were generated against the unseen 20% test set (`X_test_vec`) using `model.predict()`.

### Step 11 — Evaluation
The model was heavily evaluated. Since the dataset is imbalanced (only 12.6% spam), relying purely on "Accuracy" is misleading. Therefore, **Precision** (avoiding false positives/flagging normal mail as spam) and **Recall** (catching as much actual spam as possible) were primary indicators of success, resulting in F1-Scores. 

### Step 12 — Model Improvement
The GridSearch improved model increased the **Recall** of the spam class from 80.15% to 82.44%, while maintaining an incredibly high Precision of 99.08%. This means the improved model caught more spam messages without accidentally filtering out valid emails.

### Step 13 — Model Saving
`joblib.dump` was utilized to save the trained `vectorizer.pkl` and `spam_classifier.pkl` inside the `models/` directory so training does not need to happen on every prediction.

### Step 14 — New Message Prediction
`predict.py` loads the saved artifacts and exposes a CLI loop where users can type raw messages, which are instantly vectorized and predicted based on the saved state.

## 9. Complete Installation Instructions

1. **Clone or Navigate to the directory:**
```bash
cd spam-mail-svm
```

2. **Create a Virtual Environment (Optional but Recommended):**
```bash
python -m venv venv
```

3. **Activate the Virtual Environment:**
* **Windows:**
```bash
venv\Scripts\activate
```
* **Linux/Mac:**
```bash
source venv/bin/activate
```

4. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

## 10. How to Train the Model
To automatically fetch the dataset, clean it, train the models, generate the visualizations, and save the `.pkl` files, simply run:
```bash
python main.py
```

## 11. How to Run Prediction
To test the saved model interactively with your own messages:
```bash
python predict.py
```

## 12. Example Prediction
```text
Enter a message: Congratulations! You have won a free prize. Click here now!
Prediction: SPAM

Enter a message: Hey, are we meeting tomorrow?
Prediction: NOT SPAM
```

## 13. Results
**Baseline SVM Results (RBF Kernel, C=1.0):**
* **Accuracy:** 97.49%
* **Precision:** 100%
* **Recall:** 80.15%
* **F1-Score:** 88.98%

**Improved SVM Results (Linear Kernel, C=0.1):**
* **Accuracy:** 97.68%
* **Precision:** 99.08%
* **Recall:** 82.44%
* **F1-Score:** 90.00%

## 14. Model Comparison
The transition from an RBF to a Linear Kernel with a smaller margin penalty (`C=0.1`) reduced overfitting to the majority class. While Precision dropped by a tiny margin (from 100% to 99.08%), the model became significantly better at recognizing actual spam (Recall increased by >2%), resulting in a higher overall F1-Score of 0.9000.

## 15. Visualizations
* **`outputs/class_distribution.png`**: Displays a bar chart revealing the heavy class imbalance between Spam and Ham messages prior to splitting.
* **`outputs/confusion_matrix.png`**: Shows the True Positives, True Negatives, False Positives, and False Negatives of the baseline SVM.
* **`outputs/confusion_matrix_improved_svm.png`**: Highlights how the improved model made fewer False Negative errors.
* **`outputs/model_comparison.png`**: A grouped bar chart visually contrasting Accuracy, Precision, Recall, and F1-Score between the two trained models.

## 16. Troubleshooting
* **`Error: Trained model or vectorizer not found`**: You are trying to run `predict.py` before training the model. You must run `python main.py` first.
* **`ModuleNotFoundError`**: Ensure you have activated your virtual environment and run `pip install -r requirements.txt`.
* **Dataset Download Issues**: If the UCI URL fails, you can manually download any SMS Spam CSV dataset and place it inside the `data/` folder as `spam.csv`. Ensure the column names are exactly `Category` and `Message`.

## 17. Machine Learning Workflow
```text
Dataset (Auto Download)
  ↓
Cleaning (Drop Nulls/Duplicates)
  ↓
Train/Test Split (Stratified 80/20)
  ↓
CountVectorizer (Feature Extraction)
  ↓
SVM (Hyperparameter Tuning)
  ↓
Training
  ↓
Prediction
  ↓
Evaluation (Accuracy, Precision, Recall, F1)
  ↓
Model Saving (.pkl)
```

## 18. Future Improvements
* **TF-IDF Vectorization**: Replacing `CountVectorizer` with `TfidfVectorizer` to scale down the impact of words that appear very frequently across all messages.
* **Text Lemmatization/Stemming**: Implementing NLTK or SpaCy to reduce words to their root forms prior to vectorization.
* **API Deployment**: Wrapping the `predict.py` logic inside a FastAPI or Flask web application to serve real-time predictions over HTTP.
