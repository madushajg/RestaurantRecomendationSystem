import csv

ifile = open('zomato.csv', "r", errors='ignore')
read = csv.reader(ifile)

with open('filtered_zomato.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(
        ['Restaurant ID', 'Restaurant Name', 'Country Code', 'City',
         'Longitude', 'Latitude', 'Average Cost for two', 'Currency', 'Has Table booking',
         'Has Online delivery', 'Is delivering now', 'Switch to order menu', 'Price range', 'Aggregate rating',
         'Rating color', 'Rating text', 'Votes', 'Cuisines'])

    rowNr = 0
    for row in read:
        if rowNr >= 1:
            Restaurant_ID = row[0]
            Restaurant_Name = row[1]
            Country_Code = row[2]
            City = row[3]
            Address = row[4]
            Locality = row[5]
            Locality_Verbose = row[6]
            Longitude = row[7]
            Latitude = row[8]
            Cuisines = row[9]
            AvgCost = row[10]
            Currency = row[11]
            TableBooking = row[12]
            OnlineDelivery = row[13]
            DelNow = row[14]
            SwtchOrderMenu = row[15]
            PriceRng = row[16]
            AvgRating = row[17]
            RatingColor = row[18]
            Rating_Text = row[19]
            Votes = row[20]

            # filewriter.writerow([Restaurant_ID, Restaurant_Name,Country_Code,City,Address,Locality,Locality_Verbose,Longitude,Cuisines,AvgCost,Currency,TableBooking,OnlineDelivery,DelNow,SwtchOrderMenu,PriceRng,AvgRating,RatingColor,Rating_Text,Votes])
            filewriter.writerow(
                [Restaurant_ID, Restaurant_Name, Country_Code, City, Longitude, Latitude,
                 AvgCost, Currency, TableBooking, OnlineDelivery, DelNow, SwtchOrderMenu, PriceRng, AvgRating,
                 RatingColor, Rating_Text, Votes, Cuisines])

        rowNr = rowNr + 1
