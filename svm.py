import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("breast_cancer.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# Remove unnecessary columns
df = df.drop(["id", "Unnamed: 32"], axis=1)

# Convert diagnosis into numerical values
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Separate input and output
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create SVM model
model = SVC(kernel="linear")

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Heatmap
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap=sns.color_palette(["#2ecc71", "#e74c3c"])
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("SVM - Breast Cancer Confusion Matrix")
plt.show()