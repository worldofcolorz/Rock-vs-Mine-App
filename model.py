import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv("Sonar data.csv", header=None)

# Split data into features and target
X = data.drop(columns=60, axis=1)
y = data[60]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=1
)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
print(f"Training Accuracy: {train_acc}")
print(f"Testing Accuracy: {test_acc}")

# Save the model to a file
joblib.dump(model, 'model.pkl')
print("✅ Model saved as model.pkl")

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Save classification report
report = classification_report(y_test, model.predict(X_test), output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("classification_report.csv", index=True)

# Save confusion matrix as image
cm = confusion_matrix(y_test, model.predict(X_test))
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Rock', 'Mine'], yticklabels=['Rock', 'Mine'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

