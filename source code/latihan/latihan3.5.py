# Exercise 3.5 Naive Bayes Iris with Model Persistence (Save/Load)
import pickle
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

# 1. Load dataset Iris
X, y = load_iris(return_X_y=True)

# 2. Inisialisasi dan latih model Naive Bayes (GaussianNB)
clf = GaussianNB()
clf.fit(X, y)

# 3. MODIFIKASI EXERCISE 3.5: Simpan (Save) model ke file
filename = 'naive_bayes_iris_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(clf, file)
print(f"Model berhasil disimpan ke file '{filename}'")

# 4. MODIFIKASI EXERCISE 3.5: Muat (Load) model dari file
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)
print("Model berhasil dimuat dari file.")

# 5. Lakukan prediksi menggunakan model yang dimuat dari file
p = loaded_model.predict([[5.0, 3.4, 1.5, 0.4]])
print(f"Hasil Prediksi: {p[0]}")