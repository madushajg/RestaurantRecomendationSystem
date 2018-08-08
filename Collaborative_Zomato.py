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
restaurants = df['RestaurantName']
cities = df['City']
countries = df['CountryCode']

restaurants_arr = restaurants.values
cities_arr = cities.values
countrries_arr = countries.values

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
df.loc[index + 1] = datapoint

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

user_input = np.reshape(new, (1, -1))
restaurants_arr = np.reshape(restaurants_arr, (-1, 1))
cities_arr = np.reshape(cities_arr, (-1, 1))
# similarities_train = pairwise_distances(filtered_train, user_input, metric='cosine')
# similarities_test = pairwise_distances(filtered_test, user_input, metric='cosine')

similarities = pairwise_distances(X, user_input, metric='cosine')
# for item in filtered_test:
#     similarities.append(pairwise_distances(item, new, metric='cosine'))Yes No Yes No 3 0 French Chinese Seafood

# print(predict_class)
# print(filtered_test)
# print(similarities.shape)

final = np.concatenate((restaurants_arr, cities_arr), axis=1)
final = np.concatenate((final, similarities), axis=1)
# print(final[0][2])

maximum = final[0][2]
rest = final[0][0]

for item in final:
    if item[2] > maximum:
        maximum = item[2]
        rest = item[0]
        city =

# print(rest)
# filtered_train = np.array(filtered_train)
s = "Restaurant" + rest + " - " + citi + ""
print(len(cities_arr))

exit()