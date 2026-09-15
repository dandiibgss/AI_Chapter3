# Example 3.24 Ensemble1.py
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset Iris
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

# Inisialisasi model klasifikasi
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
clf4 = SVC()

# Inisialisasi Voting Classifier (Ensemble)
eclf = VotingClassifier(
    estimators=[
        ("lr", clf1),
        ("rf", clf2),
        ("gnb", clf3),
        ("svc", clf4),
    ],
    voting="hard",
)

# Evaluasi performa masing-masing model dan ensemble
classifiers = [clf1, clf2, clf3, clf4, eclf]
labels = ["Logistic Regression", "Random Forest", "naive Bayes", "SVM", "Ensemble"]

for clf, label in zip(classifiers, labels):
    scores = cross_val_score(clf, X, y, scoring="accuracy", cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))