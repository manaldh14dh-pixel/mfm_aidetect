
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def train_logistic_regression(self, X_train, y_train):
        print("Training Logistic Regression (Baseline)...")
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        print("Logistic Regression training complete.")
        return model

    def train_svm(self, X_train, y_train, C=1.0, kernel='rbf'):
        print(f"Training SVM (C={C}, kernel={kernel})...")
        model = SVC(
            C=C,
            kernel=kernel,
            probability=True,
            random_state=self.random_state,
            cache_size=2000
        )
        model.fit(X_train, y_train)
        print("SVM training complete.")
        return model

    def train_random_forest(self, X_train, y_train, n_estimators=200):
        print(f"Training Random Forest (n_estimators={n_estimators})...")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        print("Random Forest training complete.")
        return model

    def evaluate_model(self, model, X_val, y_val):
        print("Evaluating model...")
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_val, preds))
        return acc

    def save_model(self, model, filepath):
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")

if __name__ == "__main__":
    print("This module provides model training utilities.")
    print("Import this file to use the ModelTrainer class.")
