# Exercise 3.1 Python SVM Classifications
from sklearn import svm

# X berisi 6 sampel: [Height(cm), Weight(kg), Shoesize(UK)]
X = [
    [170, 70, 10], 
    [180, 80, 12], 
    [170, 65, 8],  
    [160, 55, 7],  
    [175, 75, 11], # Sampel tambahan 1 (Male)
    [155, 50, 6]   # Sampel tambahan 2 (Female)
]

# y berisi 6 label (0: Male, 1: Female)
y = [0, 0, 1, 1, 0, 1] 

# Inisialisasi dan latih model Support Vector Machine (SVM)
clf = svm.SVC()
clf.fit(X, y)

# Prediksi data baru
p = clf.predict([[160, 60, 7]])
print(f"Hasil prediksi: {p}")