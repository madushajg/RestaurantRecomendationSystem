import numpy as np
import pandas as pd
import matplotlib.pyplot as pyt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import SpectralClustering
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import pairwise_distances

# Get user input for HasTablebooking, HasOnlinedelivery, Isdeliveringnow, Switchtoordermenu, Pricerange, Aggregaterating and cuisine preference. Refer example_user_inputs for samples.
a_lst = []          # Start with empty list
a_lst.append("Unkown rest")
a_lst.append("Unkown_city")
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
restaurants = df['RestaurantName']
cities = df['City']

restaurants_arr = restaurants.values
cities_arr = cities.values

df.drop(['RestaurantID'], 1, inplace=True)
# df.drop(['RestaurantName'], 1, inplace=True)
df.drop(['CountryCode'], 1, inplace=True)
# df.drop(['City'], 1, inplace=True)
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

binaries = []
binaries_dict = {}

# assign 1 and o according to the users's cuisine preference.
for i in range(8, size):
    for j in range(8, len(a_lst)):
        if a_lst[j] == headers[i]:
            binaries_dict[i] = 1
    binaries.append(0)

for key, value in binaries_dict.items():
    for i in range(0, len(binaries)):
        if (i + 8) == key:
            binaries[i] = 1
            break

# the user's data point row in the row format of the the pandas data frame
datapoint = []

for i in range(0, 8):
    datapoint.append(a_lst[i])

# add the new datapoint to the dataframe to perform preprocessing
datapoint.extend(binaries)

index = df.Pricerange.count() - 1
df.loc[index + 1] = datapoint

df_alt = df
label_dictionary = []
# preform preprocessing
for column in df.columns:
    if df[column].dtype == type(object):
        le = preprocessing.LabelEncoder()
        le.fit(df[column].astype(str))
        le_mapping = dict(zip(le.transform(le.classes_), le.classes_))
        label_dictionary.append(le_mapping)
        df[column] = le.fit_transform(df[column].astype(str))  # astype casts the column vlaues to string.

X = df.values
y = df['Aggregaterating'].values
# print(X)
new = X[-1]

# # remove the new data point from the numpy arrays before dividing into test and training sets and  running it against the classifier
X = X[:-1, :]
y = y[:-1]
# divide dataset into train and test sets in 7 : 3 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# print(X_train[:, 2:])
# # print(len(y))
# print(X_train)

# derive the culsters
n_clusters = 2    # len(np.unique(y_train))
clu = SpectralClustering(n_clusters=n_clusters, n_jobs=-1)
clu.fit(X_train[:, 2:])
y_labels_train = clu.labels_
y_labels_test = clu.fit_predict(X_test[:, 2:])


predict = []
predict.append(new)
predict = np.array(predict)

# train the dataset using a classification algorithm by using the clusters derived above as the traget class/ output column
clf = MultinomialNB()
clf.fit(X_train[:, 2:], y_labels_train)
print(clf)

# predict the target class / output of the new user's datapoint
prediction = clf.predict(predict[:, 2:])
predict_class = prediction[0]
s = new.size
prediction_np = np.array(prediction)
accuracy = clf.score(X_test[:, 2:], y_labels_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

labels_train = np.zeros(shape=(1, len(y_labels_train)))
labels_train[0] = y_labels_train

labels_test = np.zeros(shape=(1, len(y_labels_test)))
labels_test[0] = y_labels_test

# add the clusters' columns to the training and test sets
X_train = np.concatenate((X_train, labels_train.T), axis=1)
X_test = np.concatenate((X_test, labels_test.T), axis=1)

# # filter out the training and test sets for datapoints that beong to the same cluster/class as the new user input datapoint
filtered_train = []
for item in X_train:
    if item[-1] == predict_class:
        filtered_train.append(item)

filtered_test = []
for item in X_test:
    if item[-1] == predict_class:
        filtered_test.append(item)

# # COLLABORATIVE FILTERING - item-to-item
filtered_train = np.array(filtered_train)
filtered_test = np.array(filtered_test)

# print(filtered_test[:, 2:])
#remove classes from the filtered data sets
filtered_train = np.delete(filtered_train, -1, axis=1)
filtered_test = np.delete(filtered_test, -1, axis=1)

# print(filtered_train.shape)

user_input = np.concatenate((new, prediction_np.T), axis=0)
user_input = np.reshape(new, (1, -1))
# print(filtered_test.shape)
similarities_train = pairwise_distances(filtered_train[:, 2:], user_input[:, 2:], metric='cosine')
similarities_test = pairwise_distances(filtered_test[:, 2:], user_input[:, 2:], metric='cosine')
# print(labels_train.T.shape)

filtered_train = np.concatenate((filtered_train, similarities_train), axis=1)
filtered_test = np.concatenate((filtered_test, similarities_test), axis=1)

# X_reshape = np.reshape(X, (X.shape[1], X.shape[0]))
# filtered_train = np.reshape(filtered_train, (filtered_train.shape[1], filtered_train.shape[0]))
# filtered_test = np.reshape(filtered_test, (filtered_test.shape[1], filtered_test.shape[0]))
# filtered_train = np.reshape(filtered_train, (152, 9534))
# print(X_reshape.shape)
label_dictionary = np.array(label_dictionary)
# Convert preprocessed dataset back to the original data valuesfiltered_train[n][i]

filtered_train = filtered_train.astype('object')
filtered_test = filtered_test.astype('object')

# for n, j in enumerate(filtered_train):
#     print(n)

for i in range(0, 6):
    for n, j in enumerate(filtered_train):
        filtered_train[n][i] = label_dictionary[i][filtered_train[n][i]]
    for m, k in enumerate(filtered_test):
        filtered_test[m][i] = label_dictionary[i][filtered_test[m][i]]

# print(filtered_test)
max = 0
recommend = filtered_train[0]

for n, j in enumerate(filtered_train):
    if filtered_train[n][-1] > max:
        max = filtered_train[n][-1]
        recommend = filtered_train[n]

max1 = 0
recommend1 = filtered_test[0]

for n, j in enumerate(filtered_test):
    if filtered_test[n][-1] > max1:
        max1 = filtered_test[n][-1]
        recommend1 = filtered_test[n]

if max1 > max:
    recommend = recommend1

s = "Restaurant - " + recommend[0] + " - " + recommend[1] + " City"
print(s)
# print(accuracy)

exit()