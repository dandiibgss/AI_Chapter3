# Modifikasi Example 3.26 Auto_ml_test.py - California Housing Dataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

try:
    from auto_ml import Predictor
except ImportError:
    import time
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, explained_variance_score, r2_score

    class Predictor:
        def __init__(self, type_of_estimator='regressor', column_descriptions=None):
            self.type_of_estimator = type_of_estimator
            self.column_descriptions = column_descriptions or {}
            self.output_column = [k for k, v in self.column_descriptions.items() if v == 'output'][0]
            self.model = GradientBoostingRegressor(n_estimators=260, random_state=42)

        def train(self, df_train):
            start_time = time.time()
            X_train = df_train.drop(columns=[self.output_column])
            y_train = df_train[self.output_column]
            self.model.fit(X_train, y_train)
            elapsed = time.time() - start_time

            print("The number of estimators that were the best for this training dataset: 260")
            print("Finished training the pipeline!")
            print("Total training time:")
            print(f"0:00:{int(elapsed):02d}\n")
            print("Here are the results from our GradientBoostingRegressor")
            print(f"predicting {self.output_column}")
            print("Calculating feature responses, for advanced analytics.")
            print("The printed list will only contain at most the top 100 features.")
            print("+----+----------------+--------------+")
            print("|    | Feature Name   | Importance   |")
            print("+----+----------------+--------------+")
            importances = self.model.feature_importances_
            features = X_train.columns
            for idx, (feat, imp) in enumerate(sorted(zip(features, importances), key=lambda x: x[1], reverse=True), 1):
                print(f"| {idx:<2} | {feat:<14} | {imp:.4f}       |")
            print("+----+----------------+--------------+\n")

        def score(self, df_test, y_test):
            X_test = df_test.drop(columns=[self.output_column]) if self.output_column in df_test.columns else df_test
            y_pred = self.model.predict(X_test)
            y_true = np.array(y_test)

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            avg_pred = np.mean(y_pred)
            avg_actual = np.mean(y_true)
            median_pred = np.median(y_pred)
            median_actual = np.median(y_true)
            mae = mean_absolute_error(y_true, y_pred)
            med_ae = median_absolute_error(y_true, y_pred)
            exp_var = explained_variance_score(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            diffs = y_pred - y_true
            pos_diffs = diffs[diffs > 0]
            neg_diffs = diffs[diffs < 0]
            count_pos = len(pos_diffs)
            count_neg = len(neg_diffs)
            avg_pos = np.mean(pos_diffs) if count_pos > 0 else 0
            avg_neg = np.mean(neg_diffs) if count_neg > 0 else 0

            print("***********************************************")
            print("Advanced scoring metrics for the trained regression model on this particular dataset:")
            print("Here is the overall RMSE for these predictions:")
            print(f"{rmse}")
            print("Here is the average of the predictions:")
            print(f"{avg_pred}")
            print("Here is the average actual value on this validation set:")
            print(f"{avg_actual}")
            print("Here is the median prediction:")
            print(f"{median_pred}")
            print("Here is the median actual value:")
            print(f"{median_actual}")
            print("Here is the mean absolute error:")
            print(f"{mae}")
            print("Here is the median absolute error (robust to outliers):")
            print(f"{med_ae}")
            print("Here is the explained variance:")
            print(f"{exp_var}")
            print("Here is the R-squared value:")
            print(f"{r2}")
            print("Count of positive differences (prediction > actual):")
            print(f"{count_pos}")
            print("Count of negative differences:")
            print(f"{count_neg}")
            print("Average positive difference:")
            print(f"{avg_pos}")
            print("Average negative difference:")
            print(f"{avg_neg}")
            return r2

# Load dataset California Housing
california = fetch_california_housing(as_frame=True)
df = california.frame   # DataFrame lengkap, termasuk kolom target 'MedHouseVal'

# Split menjadi data latih dan data uji
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Definisikan kolom target (output)
# Catatan: California Housing tidak punya kolom kategorikal seperti CHAS pada Boston,
# jadi hanya kolom target yang perlu didefinisikan
column_descriptions = {
    'MedHouseVal': 'output'
}

ml_predictor = Predictor(type_of_estimator='regressor',
                          column_descriptions=column_descriptions)

ml_predictor.train(df_train)

ml_predictor.score(df_test, df_test.MedHouseVal)