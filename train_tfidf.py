import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from prepare_dataset import prepare_dataset

DEBUG = True

def train_classifier(x_train, y_train, x_dev, y_dev):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )
    
    x_train_vec = vectorizer.fit_transform(x_train)
    x_dev_vec = vectorizer.transform(x_dev)
    
    if DEBUG:
        print(f"Train vectorized shape: {x_train_vec.shape}")
        print(f"Dev vectorized shape:   {x_dev_vec.shape}")

        print("🔹 Training Logistic Regression baseline model...")
    
    model = LogisticRegression(
        max_iter=300,
        class_weight='balanced',
        solver='liblinear'
    )
    
    model.fit(x_train_vec, y_train)
    
    y_pred = model.predict(x_dev_vec)
    
    print("\n📊 Evaluation on Dev Set")
    print("---------------------------")
    print(f"Accuracy: {accuracy_score(y_dev, y_pred):.4f}")
    print(f"F1-score: {f1_score(y_dev, y_pred, pos_label='SLI'):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_dev, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_dev, y_pred))

    return model, vectorizer     

   
def save_model(model, vectorizer, path="tfidf_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)
    print(f"💾 Model saved to {path}")


def main():
    print("🔹 Loading dataset...")
    dataset = prepare_dataset()  # dict: train/dev/test

    X_train, y_train = dataset["train"]
    X_dev, y_dev = dataset["dev"]

    model, vectorizer = train_classifier(
        X_train, y_train, 
        X_dev, y_dev
    )

    # 모델 저장 (선택)
    save_model(model, vectorizer, "tfidf_baseline.pkl")


if __name__ == "__main__":
    main()