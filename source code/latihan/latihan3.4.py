# Exercise 3.4 Python SVM Breast Cancer Histogram Plot
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset Breast Cancer
cancer = datasets.load_breast_cancer()

# Konversi dataset ke DataFrame Pandas
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

# 2. MODIFIKASI EXERCISE 3.4: Plot Histogram untuk 4 Fitur Spesifik
# Menyeleksi 4 fitur utama: Mean Radius, Mean Texture, Mean Perimeter/Size, Mean Smoothness
selected_features = [
    'mean radius', 
    'mean texture', 
    'mean perimeter',  # Mewakili 'size'
    'mean smoothness'
]

# Plot histogram untuk 4 fitur terpilih
df[selected_features].hist(bins=20, figsize=(10, 8), color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Radius, Texture, Size (Perimeter), and Smoothness', fontsize=14)
plt.tight_layout()
plt.show()

# 3. Training & Prediction dengan SVM (Sesuai Example 3.6)
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.3, random_state=109
)

# Inisialisasi dan pelatihan model SVM Linier
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluasi Model
y_pred = clf.predict(X_test)
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {metrics.precision_score(y_test, y_pred):.4f}")
print(f"Recall: {metrics.recall_score(y_test, y_pred):.4f}")