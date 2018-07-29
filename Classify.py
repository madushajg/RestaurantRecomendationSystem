import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

df = pd.read_csv('filtered_zomato.csv')

df.replace('?',-99999, inplace=True)
df.drop(['Restaurant ID'], 1, inplace=True)

Array = df.values

for column in df.columns:
    if df[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))   #astype casts the column vlaues to string.

X = np.array(df.drop(['Rating text'], 1))
y = np.array(df['Rating text'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)