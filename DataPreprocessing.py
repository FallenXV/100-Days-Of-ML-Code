import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer as imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('datasets/Data.csv')
# dataset = dataset.replace([np.nan, -np.inf], 0)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
imputer = imputer(missing_values=np.nan, strategy="mean")  # missing values like NaN are encoded as np.nan
imputer = imputer.fit(X[:, 1:3])
X = imputer.transform(X[:, 1:3])  # SimpleImputer doesn't handle NaN/inf values, these values must be removed during the imputer import

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
