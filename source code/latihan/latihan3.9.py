# Modifikasi Example 3.12: Random Forest Classification pada Diabetes Dataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load data diabetes
X, y = load_diabetes(return_X_y=True)

# Karena target diabetes bersifat kontinu (regresi), 
# kita ubah menjadi label kategori (klasifikasi) 
# dengan membagi berdasarkan median
y_class = np.where(y > np.median(y), 1, 0)  # 1 = di atas median, 0 = di bawah/sama dengan median

# Split data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.5, random_state=0
)

# Buat dan latih model Random Forest
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

# Prediksi
y_pred = clf.predict(X_test)

# Evaluasi hasil
print("Total points: %d  Correctly labeled points: %d" %
      (y_test.shape[0], (y_test == y_pred).sum()))