# Modifikasi Example 3.27 PyCaret_demo.py - Breast Cancer Dataset
import pandas as pd
from sklearn import datasets

try:
    from pycaret import classification
except (ImportError, RuntimeError):
    import time
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score, matthews_corrcoef

    class PyCaretClassificationFallback:
        def __init__(self):
            self.data = None
            self.target = None

        def setup(self, data, target, verbose=False):
            self.data = data
            self.target = target
            print("[PyCaret Fallback] Experiment Setup Completed Successfully!")

        def compare_models(self):
            X = self.data.drop(columns=[self.target])
            y = self.data[self.target]

            models = {
                'lr': ('Logistic Regression', LogisticRegression(max_iter=1000)),
                'rf': ('Random Forest Classifier', RandomForestClassifier(random_state=42)),
                'et': ('Extra Trees Classifier', ExtraTreesClassifier(random_state=42)),
                'gbc': ('Gradient Boosting Classifier', GradientBoostingClassifier(random_state=42)),
                'dt': ('Decision Tree Classifier', DecisionTreeClassifier(random_state=42)),
                'knn': ('K Neighbors Classifier', KNeighborsClassifier()),
                'nb': ('Naive Bayes', GaussianNB()),
                'svm': ('SVM - Linear Kernel', SVC(kernel='linear')),
                'ada': ('AdaBoost Classifier', AdaBoostClassifier(random_state=42)),
                'lda': ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
                'qda': ('Quadratic Discriminant Analysis', QuadraticDiscriminantAnalysis()),
            }

            results = []
            for code, (name, model) in models.items():
                start_time = time.time()
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                accs, recs, precs, f1s, kappas, mccs = [], [], [], [], [], []
                
                for train_idx, val_idx in skf.split(X, y):
                    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    model.fit(X_tr, y_tr)
                    y_pred = model.predict(X_val)
                    
                    accs.append(accuracy_score(y_val, y_pred))
                    recs.append(recall_score(y_val, y_pred, average='macro'))
                    precs.append(precision_score(y_val, y_pred, average='macro'))
                    f1s.append(f1_score(y_val, y_pred, average='macro'))
                    kappas.append(cohen_kappa_score(y_val, y_pred))
                    mccs.append(matthews_corrcoef(y_val, y_pred))
                    
                tt = time.time() - start_time
                results.append({
                    'Model': name,
                    'Accuracy': round(np.mean(accs), 4),
                    'Recall': round(np.mean(recs), 4),
                    'Prec.': round(np.mean(precs), 4),
                    'F1': round(np.mean(f1s), 4),
                    'Kappa': round(np.mean(kappas), 4),
                    'MCC': round(np.mean(mccs), 4),
                    'TT (Sec)': round(tt, 3)
                })

            res_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
            print("\n" + "="*80)
            print("PyCaret Compare Models Output (Breast Cancer Dataset)")
            print("="*80)
            print(res_df.to_string(index=False))
            print("="*80)
            return models['lr'][1]

    classification = PyCaretClassificationFallback()

# Load dataset breast cancer
cancer = datasets.load_breast_cancer(as_frame=True)
cancer.data['Target'] = cancer.target
cancer_df = cancer.data
cancer_df.head()

classification.setup(data=cancer_df, target='Target')
classification.compare_models()