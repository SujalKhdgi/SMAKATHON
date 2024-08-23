import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold

# Import the preprocessed dataset
from preprocessing import preprocess_data

def train_model():
    df = preprocess_data()

    # Define the feature columns (symptoms) and the target column (disease)
    X = df.drop(['Disease', 'Outcome Variable'], axis=1)
    y = df['Disease']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier model
    rfc = RandomForestClassifier(n_estimators=1000, random_state=42)
    rfc.fit(X_train, y_train)

    # Evaluate the model on the testing set
    y_pred = rfc.predict(X_test)
    print("Accuracy on test set:", accuracy_score(y_test, y_pred))
    print("Classification Report on test set:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix on test set:")
    print(confusion_matrix(y_test, y_pred))

    # K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
    cv_scores = cross_val_score(rfc, X, y, cv=kf)

    print("Cross-Validation Scores:", cv_scores)
    print("Mean Cross-Validation Score:", cv_scores.mean())

    # Return the trained model
    return rfc

# Uncomment and implement the prediction function if needed
# def predict_disease(symptoms, model):
#     symptoms_df = pd.DataFrame([symptoms], columns=X.columns)
#     prediction = model.predict(symptoms_df)
#     return le.inverse_transform(prediction)[0]

# Train the model
model = train_model()

# Example usage:
# symptoms = [1, 1, 0, 0, 35, 0, 1, 1]  # Replace with your input symptoms
# predicted_disease = predict_disease(symptoms, model) 
# print("Predicted disease:", predicted_disease)