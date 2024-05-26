import numpy as np
import matplotlib.pyplot as plt
import random

# Load the data and labels
data = np.load('data.npy')
labels = np.load('label.npy')

# Plot the data using a scatter plot with different colors for different labels
plt.figure(figsize=(10, 6))
for label in np.unique(labels):
    plt.scatter(data[labels == label, 0], data[labels == label, 1], label=f'Class {label}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of the Data')
plt.legend()
plt.show()


# Pseudocode of K-means clustering 

# Input: 
# - data: A dataset with n points
# - k: Number of clusters
# - max_iters: Maximum number of iterations (optional, default is 100)

# Output:
# - centroids: Final positions of the k centroids
# - labels: Cluster assignments for each data point

# Algorithm:
# 1. Randomly initialize k centroids from the data points.
# 2. Repeat until convergence or for a maximum of max_iters:
#    a. Assign each data point to the nearest centroid.
#       - For each data point:
#         - Compute the distance to each centroid.
#         - Assign the point to the centroid with the smallest distance.
#    b. Update the centroids by computing the mean of all data points assigned to each centroid.
#    c. If centroids do not change, break the loop.
# 3. Return the final centroids and the cluster assignments.




def k_means(data, k, max_iters=100):
    # Step 1: Randomly initialize k centroids from the data points
    centroids = data[random.sample(range(data.shape[0]), k)]
    
    for _ in range(max_iters):
        # Step 2a: Assign each data point to the nearest centroid
        labels = np.array([np.argmin([np.linalg.norm(point - centroid) for centroid in centroids]) for point in data])
        
        # Step 2b: Update centroids by computing the mean of all data points assigned to each centroid
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Step 2c: If centroids do not change, break the loop
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    # Step 3: Return the final centroids and the cluster assignments
    return centroids, labels

# Number of clusters (k)
k = len(np.unique(labels))

# Run K-Means
centroids, cluster_labels = k_means(data, k)

# Plot the final clustering assignments
plt.figure(figsize=(10, 6))
for i in range(k):
    plt.scatter(data[cluster_labels == i, 0], data[cluster_labels == i, 1], label=f'Cluster {i}')
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Final Clustering Assignments by K-Means')
plt.legend()
plt.show()