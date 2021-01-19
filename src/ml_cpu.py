from numpy import mean, std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
import time
from datetime import timedelta

start=time.time()

X, y = make_regression(n_samples=100000, n_features=20, n_informative=15, noise=0.1, random_state=2)

model = RandomForestRegressor()

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

delta = time.time() - start
elapsed = str(timedelta(seconds=delta))
print(f'Elapsed time : {elapsed}')