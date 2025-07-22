import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def train_model():
    # Load features and labels
    X = np.load('../outputs/features.npy')  # Make sure these exist
    y = np.load('../outputs/labels.npy')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model = SVC(kernel='linear')  # You can switch here

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save
    joblib.dump(model, '../models/rf_model.pkl')

    return model
