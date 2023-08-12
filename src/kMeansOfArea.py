import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pickle

with open("/home/ekin/Desktop/workspace/RotatetObjectDetectionReview/test_data/gt_area.pickle", 'rb') as handle:
    gt_area = pickle.load(handle)

np.sort(gt_area)
'''
plt.hist(gt_area, bins='auto', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Data')
plt.grid(True)
plt.show()
'''

# Reshape the data to have a single feature dimension
data_reshaped = np.array(gt_area).reshape(-1, 1)

# Number of clusters
num_clusters = 6

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(data_reshaped)

# Get the cluster labels
labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_
print(np.sort(cluster_centers,axis = 0))

# Plot the scatter plot
plt.scatter(range(len(gt_area)), gt_area, c=labels, cmap='viridis')
plt.xlabel('Data Point')
plt.ylabel('Value')
plt.title('K-means Clustering')
plt.savefig("/home/ekin/Desktop/workspace/RotatetObjectDetectionReview/figures/area.png")
