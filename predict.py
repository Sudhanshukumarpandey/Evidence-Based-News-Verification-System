import os
import joblib
import sys

# Import our cleaning module to prepare the input text exactly like the training data
from preprocessing import clean_text

def get_prediction(text):
    """
    Loads the saved model and vectorizer, cleans the input text, 
    vectorizes it, and predicts if it is Real or Fake news.
    
    Parameters:
    text (str): The raw news article text to predict.
    
    Returns:
    dict: A dictionary containing the prediction label, confidence score, and status.
    """
    # 1. Define file paths for the saved components
    model_path = os.path.join("saved_model", "model.pkl")
    vectorizer_path = os.path.join("saved_model", "vectorizer.pkl")
    
    # 2. Check if the saved files exist
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return {
            "status": "error",
            "message": "Model files not found. Please run train_model.py first to train and save the model."
        }
        
    # 3. Load the model and vectorizer using joblib
    # joblib.load deserializes the binary pkl files back into live Python objects
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # 4. Preprocess the input text using our standard cleaning function
    cleaned = clean_text(text)
    
    # 5. Check if the text is empty after cleaning (e.g. if the user only entered punctuation or symbols)
    if cleaned == "":
        return {
            "status": "error",
            "message": "Input text is empty or invalid after cleaning."
        }
        
    # 6. Transform the cleaned text to TF-IDF numeric features
    # NOTE: We use .transform(), NOT .fit_transform() because we must use the vocabulary
    # and IDF weights learned during the training phase.
    # We wrap the text in a list [cleaned] because the vectorizer expects an iterable (list/array) of strings.
    vectorized_text = vectorizer.transform([cleaned])
    
    # 7. Predict the class (0 = Fake, 1 = Real)
    prediction = model.predict(vectorized_text)[0]
    
    # 8. Calculate the confidence score (probability)
    # Different models have different methods for getting prediction probabilities:
    # - Logistic Regression and Naive Bayes support predict_proba()
    # - SVM (LinearSVC) does not support predict_proba() by default; instead, it uses decision_function()
    confidence = 0.0
    
    if hasattr(model, "predict_proba"):
        # predict_proba returns an array of shape (1, 2) representing [probability_of_0, probability_of_1]
        probabilities = model.predict_proba(vectorized_text)[0]
        # Get the probability corresponding to the predicted class
        confidence = probabilities[prediction]
    elif hasattr(model, "decision_function"):
        # For LinearSVC, decision_function returns a distance score from the hyper-plane boundary
        # We can map this to a pseudo-probability using a simple Sigmoid function: 1 / (1 + exp(-x))
        decision_val = model.decision_function(vectorized_text)[0]
        prob_real = 1 / (1 + os.sys.modules['math'].exp(-decision_val))
        # Confidence is prob_real if predicted 1, else (1 - prob_real)
        confidence = prob_real if prediction == 1 else (1.0 - prob_real)
        
    # Map the numeric prediction back to human-readable labels
    label_mapping = {0: "Fake", 1: "Real"}
    
    return {
        "status": "success",
        "label": label_mapping[prediction],
        "confidence": confidence,
        "cleaned_text": cleaned
    }

if __name__ == "__main__":
    # If the user runs this file directly via command line, e.g.:
    # python predict.py "Breaking: Aliens land in Central Park!"
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"your news article text here\"")
        sys.exit(1)
        
    # Get the input text from command line arguments
    input_text = sys.argv[1]
    
    print("\n--- Making Prediction ---")
    result = get_prediction(input_text)
    
    if result["status"] == "success":
        print(f"Original Text: {input_text[:100]}...")
        print(f"Prediction:    {result['label']}")
        print(f"Confidence:    {result['confidence']:.2%}")
    else:
        print(f"Error: {result['message']}")
