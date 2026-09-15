# Exercise 3.2 Python SVM Iris Classifications
from sklearn import svm, datasets

# Load dataset Iris
iris = datasets.load_iris()

# MODIFIKASI: Mengambil fitur ke-3 dan ke-4 (Petal length & Petal width)
X = iris.data[:, 2:4]  # Index 2 dan 3
y = iris.target  # 0: Setosa, 1: Versicolour, 2: Virginica

# Melatih model Support Vector Machine (SVM)
clf = svm.SVC()
clf.fit(X, y)

# Prediksi contoh data baru berdasarkan Petal length (misal: 1.5) dan Petal width (misal: 0.4)
p = clf.predict([[1.5, 0.4]])
print(f"Hasil Prediksi Kelas: {p[0]}")