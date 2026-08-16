from preprocessing import get_prepared_data
from train import train_and_evaluate
from predict import predict_message

def main():
    print("======================================================")
    print("  SPAM MAIL DETECTION USING SUPPORT VECTOR MACHINE")
    print("======================================================\n")
    
    print("Step 1: Data Preprocessing and Exploration")
    print("------------------------------------------")
    # get_prepared_data is called inside train_and_evaluate, 
    # but we can just run the full training pipeline which does everything.
    
    print("Step 2: Model Training and Evaluation")
    print("-------------------------------------")
    train_and_evaluate()
    
    print("\n======================================================")
    print("  PIPELINE COMPLETE")
    print("======================================================\n")
    
    print("Testing the saved model with sample messages...\n")
    
    spam_sample = "Congratulations! You have won a free prize. Click here now!"
    normal_sample = "Hey, are we meeting tomorrow?"
    
    print(f"Message: '{spam_sample}'")
    print(f"Prediction: {predict_message(spam_sample)}\n")
    
    print(f"Message: '{normal_sample}'")
    print(f"Prediction: {predict_message(normal_sample)}\n")
    
if __name__ == "__main__":
    main()
