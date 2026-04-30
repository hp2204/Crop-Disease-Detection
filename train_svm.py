import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================
# 1. LOAD FEATURES
# ==============================

features = np.load("features.npy")
labels = np.load("labels.npy")

print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

# ==============================
# 2. TRAIN-TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# ==============================
# 3. TRAIN SVM
# ==============================

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# ==============================
# 4. EVALUATE
# ==============================

y_pred = svm.predict(X_test)

print("\nSVM Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==============================
# 5. CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("SVM Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()
