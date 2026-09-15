# Modifikasi Example 3.28d - LazyPredict Regressor dengan Diabetes Dataset
import lazypredict
from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.1, random_state=1)

reg = LazyRegressor()
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)