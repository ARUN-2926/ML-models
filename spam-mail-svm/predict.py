import os
import joblib

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "spam_classifier.pkl")
VEC_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")

def predict_message(message):
    """
    Predicts whether a given message is SPAM or NOT SPAM using the trained model.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        print("Error: Trained model or vectorizer not found.")
        print("Please run main.py or train.py first to train and save the model.")
        return None
        
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VEC_PATH)
    except Exception as e:
        print(f"Error loading models: {e}")
        return None
        
    # Preprocess and vectorize the message
    msg_vec = vectorizer.transform([message])
    
    # Predict
    prediction = model.predict(msg_vec)[0]
    
    return "SPAM" if prediction == 1 else "NOT SPAM"

def run_interactive_prediction():
    print("\n--- Spam Message Predictor ---")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            msg = input("Enter a message: ")
            if msg.lower() in ['exit', 'quit']:
                break
                
            if not msg.strip():
                continue
                
            result = predict_message(msg)
            if result:
                print(f"Prediction: {result}\n")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
if __name__ == "__main__":
    run_interactive_prediction()
