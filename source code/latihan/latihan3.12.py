# Modifikasi Example 3.18: Multiple Linear Regression pada Linnerud Dataset
from sklearn import linear_model
from sklearn.datasets import load_linnerud
import numpy as np

# Load data Linnerud
# X = data latihan (exercise): Chins, Situps, Jumps
# y = data fisiologis (physiological): Weight, Waist, Pulse
X, y = load_linnerud(return_X_y=True)

print("Bentuk X (fitur):", X.shape)
print("Bentuk y (target):", y.shape)

# Buat dan latih model regresi linear berganda
reg = linear_model.LinearRegression()
reg.fit(X, y)

print('Coefficients: \n', reg.coef_)
print('Intercept: \n', reg.intercept_)

# Prediksi untuk data latihan baru: [Chins, Situps, Jumps]
pred = reg.predict([[5, 8, 10]])
print('Prediction (Weight, Waist, Pulse): \n', pred)