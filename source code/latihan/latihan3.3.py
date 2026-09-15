# Exercise 3.3 Python SVM Iris Scatter Plot
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset Iris dari URL (sesuai Example 3.4)
url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
df = pd.read_csv(url)

# Clean dataset dari nilai kosong (NaN)
df = df.dropna()

# 2. MODIFIKASI EXERCISE 3.3: Membuat Scatter Plot 2 Fitur Pertama
plt.figure(figsize=(8, 6))

# Mengelompokkan berdasarkan spesies agar warna titik tiap spesies berbeda
for species_name, group in df.groupby('species'):
    plt.scatter(group['sepal_length'], group['sepal_width'], label=species_name)

# Menambahkan label axis dan judul
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Scatter Plot: Sepal Length vs Sepal Width (All Data Points)')
plt.legend()
plt.grid(True)
plt.show()

# 3. Training & Prediction dengan SVM (sesuai Example 3.4)
X = df.values[:, :2]  # Mengambil 2 fitur pertama (sepal length & width)
s = df['species']
d = dict([(y, x) for x, y in enumerate(sorted(set(s)))])
y = [d[x] for x in s]

clf = svm.SVC()
clf.fit(X, y)

# Prediksi untuk sampel sepal_length = 5.4 dan sepal_width = 3.2
p = clf.predict([[5.4, 3.2]])
print(f"Hasil Prediksi Kelas: {p[0]}")