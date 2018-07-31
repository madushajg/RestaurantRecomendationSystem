import pandas as pd
import csv

df = pd.read_csv('filtered_zomato.csv')

df.replace('?',-99999, inplace=True)
df.drop(['Restaurant ID'], 1, inplace=True)


def unique(list1):
    # intilize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

    return unique_list


cuisine1 = df.Cuisine1.unique()
cuisine2 = df.Cuisine2.unique()
cuisine3 = df.Cuisine3.unique()
cuisine4 = df.Cuisine4.unique()
cuisine5 = df.Cuisine5.unique()
cuisine6 = df.Cuisine6.unique()
cuisine7 = df.Cuisine7.unique()
cuisine8 = df.Cuisine8.unique()

cuisines = []

cuisines.extend(cuisine1)
cuisines.extend(cuisine2)
cuisines.extend(cuisine3)
cuisines.extend(cuisine4)
cuisines.extend(cuisine5)
cuisines.extend(cuisine6)
cuisines.extend(cuisine7)
cuisines.extend(cuisine8)

for n, i in enumerate(cuisines):
    cuisines[n] = str(cuisines[n]).lstrip()

cuisinesUnique = unique(cuisines)

headings = ['Restaurant ID', 'Restaurant Name', 'Country Code', 'City',
         'Longitude', 'Latitude', 'Average Cost for two', 'Currency', 'Has Table booking',
         'Has Online delivery', 'Is delivering now', 'Switch to order menu', 'Price range', 'Aggregate rating',
         'Rating color', 'Rating text', 'Votes']

headings.extend(cuisinesUnique)

ifile = open('filtered_zomato.csv', "r", errors='ignore')
read = csv.reader(ifile)

with open('binarized_zomato.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(headings)

    rowNr = 0
    for row in read:
        if rowNr >= 1:
            Restaurant_ID = row[0]
            Restaurant_Name = row[1]
            Country_Code = row[2]
            City = row[3]
            Longitude = row[4]
            Latitude = row[5]
            AvgCost = row[6]
            Currency = row[7]
            TableBooking = row[8]
            OnlineDelivery = row[9]
            DelNow = row[10]
            SwtchOrderMenu = row[11]
            PriceRng = row[12]
            AvgRating = row[13]
            RatingColor = row[14]
            Rating_Text = row[15]
            Votes = row[16]

            binaries = []
            for cuisine in cuisinesUnique:
                if str(row[17]).lstrip() == cuisine or str(row[18]).lstrip() == cuisine or str(row[19]).lstrip() == cuisine or str(row[20]).lstrip() == cuisine or str(row[21]).lstrip() == cuisine \
                        or str(row[22]).lstrip() == cuisine or str(row[23]).lstrip() == cuisine or str(row[24]).lstrip() == cuisine:
                    binaries.append(1)
                else:
                    binaries.append(0)

            currentRow = [Restaurant_ID, Restaurant_Name, Country_Code, City, Longitude, Latitude,
                 AvgCost, Currency, TableBooking, OnlineDelivery, DelNow, SwtchOrderMenu, PriceRng, AvgRating,
                 RatingColor, Rating_Text, Votes]

            currentRow.extend(binaries)

            filewriter.writerow(currentRow)

        rowNr = rowNr + 1


# print(df.shape)
# df.head()
# print(len(cuisinesUnique))
