import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_excel('../data/AgriShield_Farmer_Dataset.xlsx')

# Example preprocessing
X = df.drop('Action_Label', axis=1)
y = df['Action_Label']

# Encode categorical target
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained successfully!")

# Save model and label encoder
pickle.dump(model, open('agrishield_model.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb'))

print("💾 Model and encoder saved in /model folder")