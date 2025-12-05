# compare_models.py
import pickle
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from prepare_dataset import prepare_all
from train_bert import (
    CustomBERTClassifier, 
    CustomTextClassificationDataset
)


# =====================
# Helper Functions
# =====================

def evaluate_tfidf(X_test, y_test, model_file="tfidf_baseline.pkl"):
    """Evaluate the TF-IDF baseline model on test set"""
    with open(model_file, "rb") as f:
        saved = pickle.load(f)
        model = saved["model"]
        vectorizer = saved["vectorizer"]

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="SLI"),
        "recall": recall_score(y_test, y_pred, pos_label="SLI"),
        "f1_score": f1_score(y_test, y_pred, pos_label="SLI")
    }


def evaluate_bert(X_test, y_test, model_path="bert_finetuned.pth",
                  bert_model_name="bert-base-uncased", batch_size=16, max_len=256):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    test_dataset = CustomTextClassificationDataset(X_test, y_test, tokenizer, max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = CustomBERTClassifier(bert_model_name, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preds, actuals = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predictions = torch.max(outputs, dim=1)

            preds.extend(predictions.cpu().tolist())
            actuals.extend(labels.cpu().tolist())

    return {
        "accuracy": accuracy_score(actuals, preds),
        "precision": precision_score(actuals, preds),
        "recall": recall_score(actuals, preds),
        "f1_score": f1_score(actuals, preds)
    }


# =====================
# Main Comparison
# =====================

def main():
    print("🔹 Loading dataset...")
    dataset = prepare_all()
    X_test, y_test = dataset["test"]

    print("\n🔹 Evaluating TF-IDF baseline...")
    tfidf_results = evaluate_tfidf(X_test, y_test)

    print("🔹 Evaluating BERT fine-tuned model...")
    bert_results = evaluate_bert(X_test, y_test)

    # Create a comparison table
    comparison = pd.DataFrame({
        "TF-IDF Baseline": tfidf_results,
        "BERT Fine-tuned": bert_results
    })

    print("\n📊 Performance Comparison (Test Set)")
    print("--------------------------------------")
    print(comparison)

    # Save as CSV
    comparison.to_csv("model_comparison_results.csv")
    print("\n💾 Saved comparison table to model_comparison_results.csv")


if __name__ == "__main__":
    main()
