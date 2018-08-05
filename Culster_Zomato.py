import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm, preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

# Get user input for HasTablebooking, HasOnlinedelivery, Isdeliveringnow, Switchtoordermenu, Pricerange, Aggregaterating and cuisine preference. Refer example_user_inputs for samples.
a_lst = []          # Start with empty list
s = input('Enter data: ')
a = s.split(' ')

for item in a:
    try:
        a_lst.append(int(item))
    except ValueError:
        a_lst.append(item)


# drop unnecessary columns
df = pd.read_csv('binarized_zomato.csv')
df.replace('?', -99999, inplace=True)
git
restaurants = df['RestaurantName']
cities = df['City']

restaurants_arr = restaurants.values
cities_arr = cities.values

df.drop(['RestaurantID'], 1, inplace=True)
df.drop(['RestaurantName'], 1, inplace=True)
df.drop(['CountryCode'], 1, inplace=True)
df.drop(['City'], 1, inplace=True)
df.drop(['Longitude'], 1, inplace=True)
df.drop(['Latitude'], 1, inplace=True)
df.drop(['AverageCostfortwo'], 1, inplace=True)
df.drop(['Currency'], 1, inplace=True)
df.drop(['Ratingcolor'], 1, inplace=True)
df.drop(['Ratingtext'], 1, inplace=True)
df.drop(['Votes'], 1, inplace=True)

# Get the data set columns
headers = df.dtypes.index
size = len(df.dtypes.index)
# print(a_lst)

binaries = []
binaries_dict = {}

# assign 1 and o according to the users's cuisine preference.
for i in range(6, size):
    for j in range(6, len(a_lst)):
        if a_lst[j] == headers[i]:
            binaries_dict[i] = 1
    binaries.append(0)

for key, value in binaries_dict.items():
    for i in range(0, len(binaries)):
        if (i + 6) == key:
            binaries[i] = 1
            break

# the user's data point row in the row format of the the pandas data frame
datapoint = []

for i in range(0, 6):
    datapoint.append(a_lst[i])

# add the new datapoint to the dataframe to perform preprocessing
datapoint.extend(binaries)
index = df.Pricerange.count() - 1
df.loc[index] = datapoint

# preform preprocessing
for column in df.columns:
    if df[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))  # astype casts the column vlaues to string.

X = df.values
y = df['Aggregaterating'].values
new = X[-1]

# remove the new data point from the numpy arrays before dividing into test and training sets and  running it against the classifier
X = X[:-1, :]
y = y[:-1]
# divide dataset into train and test sets in 7 : 3 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# print(len(y))

# derive the culsters
n_clusters = 5    # len(np.unique(y_train))
clu = KMeans(n_clusters=n_clusters, random_state=42)
clu.fit(X_train)
y_labels_train = clu.labels_
y_labels_test = clu.predict(X_test)


predict = []
predict.append(new)

# train the dataset using a classification algorithm by using the clusters derived above as the traget class/ output column
clf = svm.SVC()
clf.fit(X_train, y_labels_train)

# predict the target class / output of the new user's datapoint
prediction = clf.predict(predict)
predict_class = prediction[0]
s = new.size
prediction_np = np.array(prediction)

accuracy = clf.score(X_test, y_labels_test)

labels_train = np.zeros(shape=(1, len(y_labels_train)))
labels_train[0] = y_labels_train

labels_test = np.zeros(shape=(1, len(y_labels_test)))
labels_test[0] = y_labels_test

# add the clusters' columns to the training and test sets
X_train = np.concatenate((X_train, labels_train.T), axis=1)
X_test = np.concatenate((X_test, labels_test.T), axis=1)

# filter out the training and test sets for datapoints that beong to the same cluster/class as the new user input datapoint
filtered_train = []
for item in X_train:
    if item[-1] == predict_class:
        filtered_train.append(item)

filtered_test = []
for item in X_test:
    if item[-1] == predict_class:
        filtered_test.append(item)

# COLLABORATIVE FILTERING - item-to-item
filtered_train = np.array(filtered_train)
filtered_test = np.array(filtered_test)
user_input = np.concatenate((new, prediction_np.T), axis=0)
user_input = np.reshape(user_input, (1, -1))
similarities_train = pairwise_distances(filtered_train, user_input, metric='cosine')
similarities_test = pairwise_distances(filtered_test, user_input, metric='cosine')
# for item in filtered_test:
#     similarities.append(pairwise_distances(item, new, metric='cosine'))Yes No Yes No 3 0 French Chinese Seafood

# print(predict_class)
# print(filtered_test)
print(X_train[0])
# filtered_train = np.array(filtered_train)

exit()