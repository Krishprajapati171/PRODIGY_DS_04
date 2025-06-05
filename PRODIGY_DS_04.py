
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("gender_classification_v7.csv")

# Step 1: Explore
print("Data Info:\n", df.info())
print("\nValue Counts:\n", df['gender'].value_counts())

# Step 2: Visualization
plt.figure(figsize=(12, 5))

# Bar chart for gender
plt.subplot(1, 2, 1)
sns.countplot(x='gender', data=df, palette='pastel')
plt.title("Gender Distribution")

# Histogram for forehead width
plt.subplot(1, 2, 2)
sns.histplot(df['forehead_width_cm'], kde=True, bins=15, color='skyblue')
plt.title("Forehead Width Distribution")
plt.xlabel("Forehead Width (cm)")
plt.tight_layout()
plt.show()

# Step 3: Label Encoding
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])  # Male=1, Female=0

# Step 4: Define features and target
X = df.drop("gender", axis=1)
y = df["gender"]

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Step 8: Predictions and Evaluation
y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 9: Feature Importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 5))
feature_importance.sort_values().plot(kind='barh', title="Feature Importances")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
