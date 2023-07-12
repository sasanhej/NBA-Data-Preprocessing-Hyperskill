import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn import datasets


df = pd.read_csv('./data/dataset/input.txt')
dummy_regressor = DummyRegressor(strategy='quantile', quantile=0.4)
dummy_regressor.fit(X=df.X, y=df.y)
print(round(dummy_regressor.predict(df.X)[0],4))

