import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn import svm

# 1. Memuat dataset Iris dari URL
url = (
    "https://gist.githubusercontent.com/curran/"
    "a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
)
df = pd.read_csv(url)

# Inspeksi data dasar
print("Shape Data:", df.shape)
print("\n10 Data Pertama:\n", df.head(10))
print("\n10 Data Terakhir:\n", df.tail(10))
print("\nDeskripsi Statistik:\n", df.describe())

# 2. Pembersihan Data (Handling NaN)
print("\nJumlah Nilai NaN:", df.isna().sum().sum())
df = df.dropna()

# Distribusi Kelas
print("\nJumlah data per spesies:\n", df.groupby("species").size())

# 3. Visualisasi Data
# Histogram
df.hist()
pyplot.tight_layout()
pyplot.show()

# Scatter Plot Matrix
scatter_matrix(df, figsize=(10, 10))
pyplot.tight_layout()
pyplot.show()

# 4. Persiapan Fitur dan Target untuk Model
# Mengambil 2 fitur pertama (sepal_length dan sepal_width)
X = df.values[:, :2]

# Encoded label spesies dari string ke integer (0, 1, 2)
s = df["species"]
d = dict([(y, x) for x, y in enumerate(sorted(set(s)))])
y = [d[x] for x in s]

# 5. Pelatihan Model Support Vector Machine (SVM)
clf = svm.SVC()
clf.fit(X, y)

# 6. Prediksi Spesies Bunga
# Memprediksi kelas berdasarkan Sepal Length = 5.4 dan Sepal Width = 3.2
prediksi = clf.predict([[5.4, 3.2]])

# Memetakan kembali hasil prediksi angka ke nama spesies
label_map = {v: k for k, v in d.items()}
print("\nHasil Prediksi (Array Class ID):", prediksi)
print("Nama Spesies Hasil Prediksi:", label_map[prediksi[0]])