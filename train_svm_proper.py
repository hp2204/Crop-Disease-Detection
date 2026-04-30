import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =============================
# Load Features
# =============================
X_train = np.load("train_features.npy")
y_train = np.load("train_labels.npy")

X_val = np.load("val_features.npy")
y_val = np.load("val_labels.npy")

print("Data loaded successfully!")
print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# =============================
# Train SVM
# =============================
svm = SVC(kernel='rbf', C=1.0, gamma='scale')

print("Training SVM...")
svm.fit(X_train, y_train)

# =============================
# Evaluate
# =============================
print("Evaluating on validation set...")
y_pred = svm.predict(X_val)

print("\nAccuracy:", accuracy_score(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))