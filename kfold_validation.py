import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

features = np.load("features.npy")
labels = np.load("labels.npy")

svm = SVC(kernel='linear')

scores = cross_val_score(svm, features, labels, cv=5)

print("Cross Validation Scores:", scores)
print("Mean Accuracy:", scores.mean())
print("Standard Deviation:", scores.std())
