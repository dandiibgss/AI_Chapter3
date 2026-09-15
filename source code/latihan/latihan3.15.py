# Modifikasi Example 3.20: K-means Clustering dengan 3 kelompok data
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Kelompok A (fitur pertama ~1)
# Kelompok B (fitur pertama ~9-11)
# Kelompok C (fitur pertama ~5-6) -- kelompok ketiga yang baru ditambahkan
X = np.array([
    [1, 2, 3], [1, 4, 2], [1, 0, 3],      # Kelompok A
    [10, 2, 4], [9, 4, 3], [11, 0, 2],    # Kelompok B
    [5, 8, 1], [6, 9, 0], [5, 7, 2]       # Kelompok C (baru)
])

# n_clusters diubah menjadi 3 karena sekarang ada 3 kelompok data
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

print("Labels: \n", kmeans.labels_)
print("Cluster centers: \n", kmeans.cluster_centers_)
print("Prediction: \n", kmeans.predict([[12, 3, 1]]))

# Visualisasi menggunakan 2 fitur pertama (fitur ke-0 dan fitur ke-1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="viridis", s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c="red", marker="X", s=200, label="Centroids")

# Hanya SATU titik per kelompok yang diberi label (titik pertama tiap grup)
plt.annotate("Group A", (X[0, 0], X[0, 1]), textcoords="offset points", xytext=(5, 5))
plt.annotate("Group B", (X[3, 0], X[3, 1]), textcoords="offset points", xytext=(5, 5))
plt.annotate("Group C", (X[6, 0], X[6, 1]), textcoords="offset points", xytext=(5, 5))

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-means Clustering with 3 Groups")
plt.legend()
plt.grid(True)
plt.show()