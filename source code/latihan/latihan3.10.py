# Modifikasi Example 3.14: Comparison of Different Classifiers pada Diabetes Dataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

names = ["SVM", "Naive Bayes", "LDA",
         "QDA", "Decision Tree", "Random Forest",
         "Nearest Neighbors", "Neural Networks"]

classifiers = [
    SVC(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    MLPClassifier(alpha=1, max_iter=1000)]

# Load data diabetes
X, y = load_diabetes(return_X_y=True)

# Ubah target kontinu menjadi label kategori (klasifikasi) berdasarkan median
y_class = np.where(y > np.median(y), 1, 0)

# Split data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.5, random_state=0
)

# Standarisasi fitur (penting untuk SVM, KNN, Neural Network agar performa lebih stabil)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Latih dan evaluasi setiap classifier
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(name + ": " + str(score))