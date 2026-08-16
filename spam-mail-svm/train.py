import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from preprocessing import get_prepared_data

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluates the model and prints metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Not Spam (Ham)', 'Spam']))
    
    return acc, prec, rec, f1

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plots and saves the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Spam', 'Spam'], 
                yticklabels=['Not Spam', 'Spam'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    filename = "confusion_matrix.png" if "Baseline" in model_name else f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    filepath = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {filepath}")

def plot_model_comparison(baseline_metrics, improved_metrics):
    """
    Plots a comparison of baseline vs improved models.
    """
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, baseline_metrics, width, label='Baseline SVM', color='#3498db')
    rects2 = ax.bar(x + width/2, improved_metrics, width, label='Improved SVM', color='#2ecc71')
    
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim([0, 1.1])
    ax.legend(loc='lower right')
    
    # Add values on top of bars
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
                        
    filepath = os.path.join(OUTPUTS_DIR, "model_comparison.png")
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    print(f"Model comparison saved to {filepath}")

def train_and_evaluate():
    # 1. Get Data
    X, y = get_prepared_data()
    
    # 2. Train/Test Split
    # Using 20% test size and stratify to handle class imbalance
    print("\nSplitting dataset into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # 3. Text Preprocessing (Vectorization)
    print("\nVectorizing text data using CountVectorizer...")
    vectorizer = CountVectorizer(stop_words='english')
    # Fit only on training data, transform both
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Number of extracted features: {X_train_vec.shape[1]}")
    
    # 4. Baseline SVM Model
    print("\nTraining Baseline SVM (RBF Kernel, C=1.0)...")
    baseline_svm = SVC(kernel='rbf', C=1.0, random_state=42)
    baseline_svm.fit(X_train_vec, y_train)
    
    print("\nPredicting and Evaluating Baseline Model...")
    y_pred_base = baseline_svm.predict(X_test_vec)
    base_metrics = evaluate_model(y_test, y_pred_base, "Baseline SVM")
    plot_confusion_matrix(y_test, y_pred_base, "Baseline SVM")
    
    # 5. Model Improvement (Hyperparameter Tuning)
    print("\nTraining Improved SVM with GridSearchCV (this may take a minute)...")
    # Using a focused grid to keep training time reasonable
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 0.1] 
    }
    
    grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_vec, y_train)
    
    best_svm = grid_search.best_estimator_
    print(f"Best Hyperparameters found: {grid_search.best_params_}")
    
    print("\nPredicting and Evaluating Improved Model...")
    y_pred_improved = best_svm.predict(X_test_vec)
    improved_metrics = evaluate_model(y_test, y_pred_improved, "Improved SVM")
    plot_confusion_matrix(y_test, y_pred_improved, "Improved SVM")
    
    # 6. Compare Models
    plot_model_comparison(base_metrics, improved_metrics)
    
    # 7. Save Models
    print("\nSaving best model and vectorizer...")
    model_path = os.path.join(MODELS_DIR, "spam_classifier.pkl")
    vec_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
    
    joblib.dump(best_svm, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vec_path}")

if __name__ == "__main__":
    train_and_evaluate()
