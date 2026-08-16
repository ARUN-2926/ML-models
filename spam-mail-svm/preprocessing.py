import os
import pandas as pd
import urllib.request
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"
OUTPUTS_DIR = "outputs"
CSV_PATH = os.path.join(DATA_DIR, "spam.csv")

def download_and_prepare_dataset():
    """
    Downloads the SMS Spam Collection dataset from UCI if it doesn't exist,
    and prepares it into a clean spam.csv file with 'Category' and 'Message' columns.
    """
    if os.path.exists(CSV_PATH):
        print(f"Dataset already exists at {CSV_PATH}")
        return pd.read_csv(CSV_PATH)
        
    print("Downloading SMS Spam Collection dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = os.path.join(DATA_DIR, "smsspamcollection.zip")
    
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete. Extracting...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
        
    extracted_file = os.path.join(DATA_DIR, "SMSSpamCollection")
    
    print("Processing and saving dataset to CSV...")
    # The dataset is tab-separated with no header
    df = pd.read_csv(extracted_file, sep='\t', header=None, names=['Category', 'Message'])
    
    # Save as CSV with correct headers
    df.to_csv(CSV_PATH, index=False)
    
    # Clean up downloaded raw files
    os.remove(zip_path)
    os.remove(extracted_file)
    if os.path.exists(os.path.join(DATA_DIR, "readme")):
        os.remove(os.path.join(DATA_DIR, "readme"))
        
    print(f"Dataset saved to {CSV_PATH}")
    return df

def explore_and_clean_data(df):
    """
    Performs data exploration, cleaning, and generates visualizations.
    """
    print("\n--- Data Exploration ---")
    print(f"Number of rows and columns: {df.shape}")
    print(f"Column names: {list(df.columns)}")
    
    print("\nDataset Info:")
    df.info()
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Drop missing values if any
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        print(f"Shape after dropping missing values: {df.shape}")
        
    print("\nDuplicate Values:")
    duplicates = df.duplicated().sum()
    print(f"Found {duplicates} duplicate rows.")
    
    if duplicates > 0:
        df = df.drop_duplicates(keep='first')
        print(f"Shape after dropping duplicates: {df.shape}")
        
    # Class distribution
    spam_count = df[df['Category'] == 'spam'].shape[0]
    ham_count = df[df['Category'] == 'ham'].shape[0]
    
    print(f"\nNumber of spam messages: {spam_count}")
    print(f"Number of non-spam (ham) messages: {ham_count}")
    print(f"Class distribution: \n{df['Category'].value_counts(normalize=True) * 100}")
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Category', palette=['#2ecc71', '#e74c3c'])
    plt.title('Distribution of Spam vs Non-Spam Messages')
    plt.xlabel('Category')
    plt.ylabel('Count')
    
    viz_path = os.path.join(OUTPUTS_DIR, "class_distribution.png")
    plt.savefig(viz_path, bbox_inches='tight')
    plt.close()
    print(f"Class distribution visualization saved to {viz_path}")
    
    return df

def get_prepared_data():
    """
    Main preprocessing function to get clean data for training.
    """
    df = download_and_prepare_dataset()
    df_clean = explore_and_clean_data(df)
    
    # Encode categories: spam=1, ham=0
    df_clean['Label'] = df_clean['Category'].map({'spam': 1, 'ham': 0})
    
    return df_clean['Message'], df_clean['Label']

if __name__ == "__main__":
    get_prepared_data()
