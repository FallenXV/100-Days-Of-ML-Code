import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer as imputer

dataset = pd.read_csv('datasets/Data.csv')
# dataset = dataset.replace([np.nan, -np.inf], 0)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
imputer = imputer(missing_values=np.nan, strategy="mean")  # missing values like NaN are encoded as np.nan
imputer = imputer.fit(X[:, 1:3])
X = imputer.transform(X[:, 1:3])  # SimpleImputer doesn't handle NaN/inf values, these values must be removed during
