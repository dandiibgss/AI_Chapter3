# Exercise 3.7 Principal Component Analysis Breast Cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import datasets

# 1. MODIFIKASI EXERCISE 3.7: Load dataset Breast Cancer
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

# 2. Plot Data Original (menggunakan 2 fitur pertama: Mean Radius & Mean Texture)
f = plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')
plt.title('Original Breast Cancer Data (2 Features)')

# 3. Perform PCA (Mereduksi dimensi data menjadi 3 komponen utama)
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X1 = pca.transform(X)

# 4. Plot PCA Data (menampilkan 2 komponen utama pertama: PCA1 & PCA2)
g = plt.figure(2)
plt.scatter(X1[:, 0], X1[:, 1], c=y, cmap='viridis')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA Breast Cancer Data')

plt.show()