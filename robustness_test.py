import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load features
features = np.load("features.npy")
labels = np.load("labels.npy")

# Add Gaussian noise
noise = np.random.normal(0, 0.1, features.shape)
noisy_features = features + noise

# Split
X_train, X_test, y_train, y_test = train_test_split(
    noisy_features, labels, test_size=0.2, random_state=42
)

# Train
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Evaluate
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy with Noise:", accuracy)
