# Modifikasi Example 3.20: K-means Clustering dengan make_blobs()
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate data sampel menggunakan make_blobs()
X, y_true = make_blobs(
    n_samples=300,      # jumlah titik data
    centers=3,          # jumlah cluster/pusat data yang dibuat
    n_features=2,        # jumlah fitur (dimensi)
    cluster_std=1.0,     # tingkat sebaran/deviasi tiap cluster
    random_state=0
)

# Buat dan latih model K-means
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

print("Labels: \n", kmeans.labels_)
print("Cluster centers: \n", kmeans.cluster_centers_)

# Prediksi cluster untuk titik data baru
pred = kmeans.predict([[0, 2]])
print("Prediction for new point [0, 2]: \n", pred)

# Visualisasi hasil clustering
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="viridis", label="Data points")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c="red", marker="X", s=200, label="Centroids")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-means Clustering with make_blobs()")
plt.legend()
plt.grid(True)
plt.show()