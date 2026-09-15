import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import lazypredict
from lazypredict.Supervised import LazyRegressor

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
# Subsample dataset for fast execution
X, y = X[:1000], y[:1000]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)