# Modifikasi Example 3.20: K-means Clustering dengan tambahan titik data
from sklearn.cluster import KMeans
import numpy as np

# Kelompok A (dekat x=1) ditambah 2 titik baru: [0,1,4] dan [2,3,2]
# Kelompok B (dekat x=9-11) ditambah 2 titik baru: [9,1,2] dan [10,3,5]
X = np.array([
    [1, 2, 3], [1, 4, 2], [1, 0, 3], [0, 1, 4], [2, 3, 2],      # Kelompok A (5 titik)
    [10, 2, 4], [9, 4, 3], [11, 0, 2], [9, 1, 2], [10, 3, 5]    # Kelompok B (5 titik)
])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

print("Labels: \n", kmeans.labels_)
print("Cluster centers: \n", kmeans.cluster_centers_)
print("Prediction: \n", kmeans.predict([[12, 3, 1]]))