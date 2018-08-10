# RestaurantRecomendationSystem

In this machine learining project, We have to use the free data set collected by Zomato, the restaurant review platform which contains data from a number of restaurants around the world. We propose an analysis where we analyze the cuisine types  (eg: Chinese, Thai, Indian, Western etc.) and the costs in each country/region and the classifier learns the the trends in each of the countries/regions. Then depending on the user's travel destinations, cuisine type preference and the budget that he/she can afford, we propose to recommend the restaurants that best suits the user at that location.



## Pre Requirements

* Python 3.6 or higher
* NumPy 1.8.2 or higher
* SciPy 0.13.3 or higher
* Sk-learn 0.18 or higher

* Rapid miner 8.2.1


If you already have a working installation of numpy and scipy, the easiest way to install scikit-learn is using pip

`pip install -U scikit-learn`

or conda:

`conda install scikit-learn`

For the virtual environment you can use anaconda virtual environment or your choice.


##Results

The recommendation from the K-Means clustering using the Cosine and Euclidean
similarity was unsatisfactory.
* The approach of using XGBoost for clustering followed by classification by SVM and Cosine
collaborative filtering was also unsatisfactory.
* The approach to use collaborative cosine similarity filtering alone also did not yield the
required recommendations.
* However, using the Manhattan distance for the collaborative filtering of DBSCAN clusters
returned satisfactory results. Also, the results remained unchanged when the classifier was
changed (Decision trees, Naïve Bayes, SVM and KNN).

##Conclusion
   In the above analysis, out of all the approaches tried, the best recommendations were
made by the DBSCAN clustering technique filtered using the Manhattan distance, with a few
deviations.
    The reason for the deviations in the other approaches may be due to the dataset. Many of
the available features being categorical in nature may have contributed for inaccuracies. In the
future the objective is to use a better data set and to use classification approaches with a suitable
ground truth to perform better recommendations.