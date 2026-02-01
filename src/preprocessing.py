import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.impute import SimpleImputer
from data_loader import data_titanic
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


df = data_titanic()
target = 'Survived'

df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

X = df.drop(columns=[target])
Y = df[target]

age_imputer = SimpleImputer(strategy='median')
X[['Age']] = age_imputer.fit_transform(X[['Age']])

embarked_imputer = SimpleImputer(strategy='most_frequent')
X[['Embarked']] = embarked_imputer.fit_transform(X[['Embarked']])

X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

enconder = OneHotEncoder(sparse_output=False, drop = 'first')

embarked_encoded = enconder.fit_transform(X[['Embarked']])
embarked_df = pd.DataFrame(
    embarked_encoded,
    columns=enconder.get_feature_names_out(['Embarked']),
    index= X.index
)

X = pd.concat([X, embarked_df], axis=1)
X = X.drop(columns=['Embarked'])

