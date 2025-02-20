import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
from transformers import BertTokenizer, BertModel
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from collections import Counter

# Load dataset
data = pd.read_csv("dataset.csv")

# Ensure 'Anomaly' column exists
if 'Anomaly' not in data.columns:
    print("❌ Error: 'Anomaly' column missing from dataset. Ensure dataset.csv contains this column.")
    exit()

# Handle missing values and duplicates
data.dropna(subset=['Text'], inplace=True)
data.drop_duplicates(subset=['Text'], inplace=True)
data['Text'] = data['Text'].apply(lambda x: '[UNK]' if pd.isna(x) else x)

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate BERT embeddings
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Generate embeddings
data['Embeddings'] = data['Text'].apply(lambda x: get_bert_embeddings(x))
embeddings = np.vstack(data['Embeddings'].values)

# Print class distribution before resampling
print("🔹 Class distribution before undersampling:", data['Anomaly'].value_counts())

# Visualization 1: Class Distribution Before Undersampling
plt.figure(figsize=(6, 4))
sns.countplot(x=data['Anomaly'], palette="coolwarm")
plt.xlabel("Class Label (1 = Normal, -1 = Anomaly)")
plt.ylabel("Count")
plt.title("Class Distribution Before Undersampling")
plt.show()

# Fix undersampling issue dynamically
class_counts = Counter(data['Anomaly'])
min_class = min(class_counts.values())  # Find the minority class count

# Ensure a valid sampling strategy
if min_class < 2:  # Avoid issues if there are too few anomalies
    print("⚠️ Warning: Not enough samples in one class for undersampling. Skipping resampling.")
    X_resampled, y_resampled = embeddings, data['Anomaly'].values  # Use original dataset
else:
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = rus.fit_resample(embeddings, data['Anomaly'])

# Print class distribution after resampling
print("🔹 Class distribution after resampling:", Counter(y_resampled))

# Visualization 2: Class Distribution After Undersampling
plt.figure(figsize=(6, 4))
sns.countplot(x=y_resampled, palette="coolwarm")
plt.xlabel("Class Label (1 = Normal, -1 = Anomaly)")
plt.ylabel("Count")
plt.title("Class Distribution After Undersampling")
plt.show()

# Ensure train-test split contains both classes
if len(set(y_resampled)) > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
else:
    print("❌ Error: Not enough anomalies. Adjust dataset balancing.")
    exit()

# Train Isolation Forest for Anomaly Detection
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X_train)

# Train Random Forest for Performance Comparison
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Check class distribution to confirm balance
print("🔹 Training Set Distribution:", Counter(y_train))
print("🔹 Test Set Distribution:", Counter(y_test))

# Model Evaluation - Random Forest
print("\n📊 Classification Report (Random Forest):")
print(classification_report(y_test, rf_predictions, zero_division=1))

# Visualization 3: Confusion Matrix
plt.figure(figsize=(6, 5))
conf_matrix = confusion_matrix(y_test, rf_predictions)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Visualization 4: PCA 2D Scatter Plot of BERT Embeddings
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(X_resampled)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=y_resampled, palette={1: "blue", -1: "red"})
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Visualization of BERT Embeddings")
plt.legend(title="Class", labels=["Normal", "Anomaly"])
plt.show()

# Visualization 5: Feature Importance from Random Forest
feature_importance = rf_model.feature_importances_
plt.figure(figsize=(8, 6))
plt.plot(feature_importance, marker="o")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.show()

# Visualization 6: ROC-AUC Curve (Only if y_test has both classes)
if len(set(y_test)) > 1:
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Anomaly Detection")
    plt.legend(loc="lower right")
    plt.show()
else:
    print("⚠️ Skipping ROC Curve: Only one class present in y_test.")

# Save trained Isolation Forest model
with open('bert_anomaly_model.pkl', 'wb') as f:
    pickle.dump(iso_forest, f)

print("✅ Model training and visualization completed successfully!")
