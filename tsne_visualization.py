import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

features = np.load("features.npy")
labels = np.load("labels.npy")

# Use subset (t-SNE is heavy)
subset_size = 3000
features_subset = features[:subset_size]
labels_subset = labels[:subset_size]

tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(features_subset)

plt.figure()
plt.scatter(reduced[:,0], reduced[:,1], c=labels_subset)
plt.title("t-SNE Visualization of CNN Features")
plt.show()
