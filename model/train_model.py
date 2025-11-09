import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.utils import resample
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import numpy as np
import pickle
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ---------------------------
# Path Setup
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "AgriShield_Farmer_Dataset.xlsx")
MODEL_DIR = os.path.join(BASE_DIR, "model")
REPORT_PATH = os.path.join(MODEL_DIR, "model_report.txt")

os.makedirs(MODEL_DIR, exist_ok=True)

print("📂 Loading dataset from:", DATA_PATH)

# ---------------------------
# Load dataset
# ---------------------------
try:
    df = pd.read_excel(DATA_PATH)
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Dataset not found at: {DATA_PATH}")

# ---------------------------
# Select relevant numeric features (matches Flask inputs)
# ---------------------------
FEATURES = ['Rainfall', 'Temperature', 'Soil_pH', 'Humidity', 'Financial_Score']

missing = [col for col in FEATURES if col not in df.columns]
if missing:
    raise ValueError(f"❌ Missing expected columns in dataset: {missing}")

X = df[FEATURES]
y = df['Action_Label']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ---------------------------
# Balance Classes (Upsample)
# ---------------------------
data_combined = pd.concat([X, pd.Series(y_encoded, name='Target')], axis=1)
max_size = data_combined['Target'].value_counts().max()
balanced_data = pd.DataFrame()

for class_index, group in data_combined.groupby('Target'):
    balanced_group = resample(group, replace=True, n_samples=max_size, random_state=42)
    balanced_data = pd.concat([balanced_data, balanced_group])

X = balanced_data.drop('Target', axis=1)
y_encoded = balanced_data['Target']

print("\n🔎 Label distribution after balancing:")
print(pd.Series(y_encoded).value_counts())

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"📊 Training samples: {len(X_train)} | Test samples: {len(X_test)}")

# ---------------------------
# Train Optimized Random Forest
# ---------------------------
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ---------------------------
# Evaluate Model
# ---------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='accuracy')
print(f"📈 Cross-validation Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Final Test Accuracy: {acc*100:.2f}%")

labels_present = unique_labels(y_test, y_pred)
label_names = le.inverse_transform(labels_present)
report = classification_report(
    y_test, y_pred, labels=labels_present, target_names=label_names, zero_division=0
)
print("\n📊 Classification Report:")
print(report)

# ---------------------------
# Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_test, y_pred, labels=labels_present)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(cmap=plt.cm.Greens, xticks_rotation='vertical')
plt.title("Confusion Matrix - AgriShield (Final Model)")
plt.tight_layout()
plt.show()

# ---------------------------
# 🌾 Feature Importance Visualization
# ---------------------------
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': np.round(feature_importance, 3)
}).sort_values(by='Importance', ascending=False)

print("\n🌾 Feature Importance:")
print(importance_df)

plt.figure(figsize=(8, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='green')
plt.gca().invert_yaxis()
plt.title("🌾 AgriShield Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# ---------------------------
# Save Model and Report
# ---------------------------
model_path = os.path.join(MODEL_DIR, "agrishield_model.pkl")
encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)
with open(encoder_path, "wb") as f:
    pickle.dump(le, f)

with open(REPORT_PATH, "a", encoding="utf-8") as f:
    f.write("============================================\n")
    f.write(f"📅 Training run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"🎯 Accuracy: {acc*100:.2f}%\n")
    f.write(f"📈 Cross-Validation: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%\n")
    f.write("📊 Classification Report:\n")
    f.write(report)
    f.write("\n🌾 Feature Importance:\n")
    f.write(importance_df.to_string(index=False))
    f.write("\n============================================\n\n")

print(f"\n💾 Model saved to: {model_path}")
print(f"💾 Label encoder saved to: {encoder_path}")
print(f"🧾 Model report saved to: {REPORT_PATH}")
print("🎯 Training process completed successfully!")
