import os
import pandas as pd
import numpy as np
import joblib

# Import train_test_split to divide data into training and testing sets
from sklearn.model_selection import train_test_split, cross_val_score

# Import the vectorizer to convert cleaned text to numerical values
from sklearn.feature_extraction.text import TfidfVectorizer

# Import classical Machine Learning classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Import performance evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

# Import our custom text cleaning function
from preprocessing import clean_text

def load_and_prepare_data():
    """
    Loads True.csv and Fake.csv, cleans empty values, adds label column,
    merges them, and shuffles the final dataset.
    
    Returns:
    pd.DataFrame: Merged and shuffled dataframe containing news data.
    """
    print("Step 2.1: Loading datasets...")
    
    # Define file paths
    true_path = os.path.join("dataset", "True.csv")
    fake_path = os.path.join("dataset", "Fake.csv")
    
    # Read CSV files using pandas
    # We use try-except to handle cases where the user hasn't downloaded the files yet
    try:
        df_true = pd.read_csv(true_path)
        df_fake = pd.read_csv(fake_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find dataset files in 'dataset/' folder. {e}")
        print("Please make sure True.csv and Fake.csv exist in the dataset/ directory.")
        return None

    # Step 2.2: Remove Reuters prefix tags from true news to prevent source bias/leakage
    # True news in the ISOT dataset contains "CITY (Reuters) - " or "Reuters - " source tags.
    # If we don't remove these, the model learns that any article containing "Reuters" is real,
    # causing it to classify other real news articles (which lack this tag) as fake.
    import re
    def strip_reuters_prefix(text):
        if not isinstance(text, str):
            return ""
        # Strip "ANYTHING (Reuters) - " or "Reuters - " from the start of the text
        text_cleaned = re.sub(r'^.*?\(Reuters\)\s*[-–]+\s*', '', text)
        text_cleaned = re.sub(r'^Reuters\s*[-–]+\s*', '', text_cleaned)
        return text_cleaned
        
    print("Step 2.2: Stripping source metadata tags to prevent model leakage...")
    df_true['text'] = df_true['text'].apply(strip_reuters_prefix)

    # Step 2.3: Add label columns
    # 1 represents Real news, 0 represents Fake news
    df_true['label'] = 1
    df_fake['label'] = 0
    
    # Step 2.4: Combine the two dataframes
    # axis=0 means we stack them vertically (row-wise)
    df_combined = pd.concat([df_true, df_fake], axis=0, ignore_index=True)
    
    # Step 2.4: Handle missing values (Crucial for real dataset robustness!)
    # Fill any NaN (Null) values in title or text columns with empty strings
    df_combined['title'] = df_combined['title'].fillna('')
    df_combined['text'] = df_combined['text'].fillna('')
    
    # Step 2.5: Combine title and text into a single training feature
    # Headlines (titles) often contain critical patterns (like uppercase clickbaits)
    df_combined['full_text'] = df_combined['title'] + " " + df_combined['text']
    
    # Step 2.6: Shuffle the dataset
    # frac=1 means shuffle 100% of the data
    # random_state=42 is a seed for reproducibility (so we get the same shuffle every time we run it)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Dataset successfully loaded. Total rows: {len(df_combined)}")
    print(f"Real articles: {len(df_combined[df_combined['label'] == 1])}")
    print(f"Fake articles: {len(df_combined[df_combined['label'] == 0])}")
    
    return df_combined

def main():
    # 1. Load and prepare data
    df = load_and_prepare_data()
    if df is None:
        return
    
    # 2. Text Preprocessing
    print("\nStep 3: Cleaning text (lowercasing, punctuation, URLs, numbers, stopwords, lemmatization)...")
    # We apply our custom cleaning function on 'full_text' column
    # On real datasets with 40,000+ rows, this can take a couple of minutes.
    # We print a message so the user knows the script is running.
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    
    # Drop rows where cleaned_text is empty to avoid empty strings in training
    df = df[df['cleaned_text'] != ''].reset_index(drop=True)
    
    # 3. Split dataset
    # X contains the features (independent variable: text)
    # y contains the target label (dependent variable: label 0 or 1)
    X = df['cleaned_text']
    y = df['label']
    
    # Split into 80% train and 20% test
    # random_state makes sure split is identical each time we run it
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Feature Extraction (TF-IDF Vectorization)
    print("\nStep 4: Converting text to numerical features using TF-IDF...")
    # max_features=5000: limits vocabulary size to top 5000 words to save memory & prevent overfitting
    # ngram_range=(1,2): includes unigrams (single words) and bigrams (pairs of words)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    # Learn vocabulary and IDF from training data and transform to matrix
    X_train_vectorized = vectorizer.fit_transform(X_train)
    # Transform test data using the already learned vocabulary (avoids Data Leakage!)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # 5. Define ML Models
    # We train Logistic Regression, Multinomial Naive Bayes, and Linear Support Vector Classifier (SVM)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=42, dual=False)
    }
    
    best_accuracy = 0
    best_model_name = ""
    best_model = None
    
    comparison_results = []
    
    print("\nStep 5 & 6: Training models and comparing metrics...")
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        # Fit model on training data
        model.fit(X_train_vectorized, y_train)
        
        # Predict on testing data
        y_pred = model.predict(X_test_vectorized)
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate Cross-Validation Score
        # We use a dynamic number of folds based on class counts to avoid crashing on tiny datasets
        # min_class_count gets the frequency of the less frequent class in the training data
        min_class_count = y_train.value_counts().min()
        if min_class_count >= 2:
            n_folds = min(5, min_class_count)
            cv_scores = cross_val_score(model, X_train_vectorized, y_train, cv=n_folds, scoring='accuracy')
            mean_cv = np.mean(cv_scores)
        else:
            mean_cv = 0.0  # Not enough data to perform cross-validation
        
        # Save metrics for summary comparison
        comparison_results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "CV Accuracy": mean_cv
        })
        
        # Print metrics
        print(f"Accuracy:      {accuracy:.4f}")
        print(f"Precision:     {precision:.4f} (Ability to not label a fake article as real)")
        print(f"Recall:        {recall:.4f} (Ability to find all real articles)")
        print(f"F1 Score:      {f1:.4f} (Harmonic mean of precision and recall)")
        print(f"Cross-Val Acc: {mean_cv:.4f} (Average accuracy across 5 folds)")
        
        # Print Confusion Matrix
        # Format:
        # [ [True Negative (Fake predicted as Fake), False Positive (Fake predicted as Real)],
        #   [False Negative (Real predicted as Fake), True Positive (Real predicted as Real)] ]
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Print Classification Report (Precision, Recall, F1 for each class)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Keep track of the model with the highest test accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = model
            
    # 6. Print Model Comparison Table
    print("\n=======================================================")
    print("MODEL COMPARISON SUMMARY")
    print("=======================================================")
    print(f"{'Model Name':<25} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<6} | {'F1 Score':<8} | {'CV Score':<8}")
    print("-" * 75)
    for res in comparison_results:
        print(f"{res['Model']:<25} | {res['Accuracy']:.4f}   | {res['Precision']:.4f}    | {res['Recall']:.4f} | {res['F1 Score']:.4f}   | {res['CV Accuracy']:.4f}")
    print("=======================================================")
    
    # 7. Save the Best Model
    print(f"\nStep 7: Saving the best model ({best_model_name}) and vectorizer...")
    # Ensure saved_model directory exists
    os.makedirs("saved_model", exist_ok=True)
    
    # Serialize and save model and vectorizer
    joblib.dump(best_model, os.path.join("saved_model", "model.pkl"))
    joblib.dump(vectorizer, os.path.join("saved_model", "vectorizer.pkl"))
    
    # Save comparison stats to a text file for frontend visualization later
    stats = {
        "best_model": best_model_name,
        "best_accuracy": best_accuracy,
        "results": comparison_results
    }
    joblib.dump(stats, os.path.join("saved_model", "model_comparison_stats.pkl"))
    
    print("Model and Vectorizer saved successfully in saved_model/")
    print(f"Best Model Saved: {best_model_name} with Accuracy {best_accuracy:.4f}")

if __name__ == "__main__":
    main()
