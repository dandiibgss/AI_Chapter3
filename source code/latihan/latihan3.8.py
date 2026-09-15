# Exercise 3.8 Decision Tree Wine Classification
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. MODIFIKASI EXERCISE 3.8: Mengganti load_iris dengan load_wine
X, y = load_wine(return_X_y=True)

# 2. Membagi dataset menjadi Data Latih (50%) dan Data Uji (50%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0
)

# 3. Inisialisasi dan melatih model Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# 4. Evaluasi Performa Model
y_pred = clf.predict(X_test)
N = y_test.shape[0]
C = (y_test == y_pred).sum()

print("Total points: %d Correctly labeled points : %d" % (N, C))
print("Accuracy: %.2f%%" % ((C / N) * 100))