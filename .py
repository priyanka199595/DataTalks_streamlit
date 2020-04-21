import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
#import plotly.figure_factory as ff
import time


from datetime import datetime
import seaborn as sns
sns.set_style("whitegrid")
import os
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import numexpr as ne

import xgboost as xgb
from surprise import Reader, Dataset
from surprise import BaselineOnly
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise.model_selection import GridSearchCV

page = st.sidebar.selectbox("Choose a page", ["Visualization","Homepage", "EDA", "About","Documentation"])

st.sidebar.title("Movie Rating Survey :smile:")
 
Movie = st.sidebar.multiselect("Which do you like the most?",
                            ("Avengers","The Golden Compass","Harry Potter"))
 
director = st.sidebar.multiselect("Who is you fav director?",
                            ("Rv","Av","MP"))
 
zonors = st.sidebar.multiselect("which topic you love",
                            ("Action","Horror","Thriller"))




st.title('DataTalks')
st.write("Here **powerful and interactive graphics** will help you to precisely analyse :sunglasses:")

if not os.path.isfile("Data/NetflixRatings.csv"):
    startTime = datetime.now()
    data = open("Data/NetflixRatings.csv", mode = "w") 
    files = ['Data/combined_data_4.txt']
    for file in files:
        print("Reading from file: "+str(file)+"...")
        with open(file) as f:  
            for line in f:
                line = line.strip() 
                if line.endswith(":"):
                    movieID = line.replace(":", "")
                else:
                    row = [] 
                    row = [x for x in line.split(",")] #custID, rating and date are separated by comma
                    row.insert(0, movieID)
                    data.write(",".join(row))
                    data.write("\n")
        print("Reading of file: "+str(file)+" is completed\n")
    data.close()
    print("Total time taken for execution of this code = "+str(datetime.now() - startTime))

else:
    print("data is already loaded")


# creating data frame from our output csv file.
if not os.path.isfile("Data/NetflixData.pkl"):
    startTime = datetime.now()
    Final_Data = pd.read_csv("Data/NetflixRatings.csv", sep=",", names = ["MovieID","CustID", "Ratings", "Date"])
    Final_Data["Date"] = pd.to_datetime(Final_Data["Date"])
    Final_Data.sort_values(by = "Date", inplace = True)
    print("Time taken for execution of above code = "+str(datetime.now() - startTime))
    st.write("data frame created")
else:
    print("data frame already present")

# storing pandas dataframe as a picklefile for later use
if not os.path.isfile("Data/NetflixData.pkl"):
    Final_Data.to_pickle("Data/NetflixData.pkl")
    st.write("pkl created")
else:
    Final_Data = pd.read_pickle("Data/NetflixData.pkl")
    print("pkl already present")

if st.checkbox("Show Final_Data"):
    st.write(Final_Data)
    if st.checkbox("Show all the column Names"):
        st.write(Final_Data.columns)
    
if st.checkbox("Show size of dataset"):
    if st.checkbox("Show row size"):
        st.write(Final_Data.shape[0])
    if st.checkbox("Show column size"):
        st.write(Final_Data.shape[1])
    if st.checkbox("Show complete dataset size"):
        st.write(Final_Data.shape)
    if st.checkbox("Show desc of Ratings in final data"):
        Final_Data.describe()["Ratings"]    

st.write("**displaying final dataset header lines using area chart**")
st.area_chart(Final_Data)


print("Number of NaN values = "+str(Final_Data.isnull().sum()))

duplicates = Final_Data.duplicated(["MovieID","CustID", "Ratings"])
print("Number of duplicate rows = "+str(duplicates.sum()))

if st.checkbox("Show unique customer & movieId in Total Data:"):
    st.write("Total number of movie ratings = ", str(Final_Data.shape[0]))
    st.write("Number of unique users = ", str(len(np.unique(Final_Data["CustID"]))))
    st.write("Number of unique movies = ", str(len(np.unique(Final_Data["MovieID"]))))

if not os.path.isfile("Data/TrainData.pkl"):
    Final_Data.iloc[:int(Final_Data.shape[0]*0.80)].to_pickle("Data/TrainData.pkl")
    Train_Data = pd.read_pickle("Data/TrainData.pkl")
    Train_Data.reset_index(drop = True, inplace = True)
else:
    Train_Data = pd.read_pickle("Data/TrainData.pkl")
    Train_Data.reset_index(drop = True, inplace = True)

if not os.path.isfile("Data/TestData.pkl"):
    Final_Data.iloc[int(Final_Data.shape[0]*0.80):].to_pickle("Data/TestData.pkl")
    Test_Data = pd.read_pickle("Data/TestData.pkl")
    Test_Data.reset_index(drop = True, inplace = True)
else:
    Test_Data = pd.read_pickle("Data/TestData.pkl")
    Test_Data.reset_index(drop = True, inplace = True)

if st.checkbox("Showing dataset of Train_Data & Test_Data"):
    st.area_chart(Train_Data)
    st.area_chart(Test_Data)

if st.checkbox("Show unique customer & movieId in Train DataSet:"):
    st.write("Total number of movie ratings in train data = ", str(Train_Data.shape[0]))
    st.write("Number of unique users in train data = ", str(len(np.unique(Train_Data["CustID"]))))
    st.write("Number of unique movies in train data = ", str(len(np.unique(Train_Data["MovieID"]))))
    st.write("Highest value of a User ID = ", str(max(Train_Data["CustID"].values)))
    st.write("Highest value of a Movie ID =  ", str(max(Train_Data["MovieID"].values)))


if st.checkbox("Show unique customer & movieId in Test DataSet:"):
    st.write("Total number of movie ratings in Test data = ", str(Test_Data.shape[0]))
    st.write("Number of unique users in Test data = ", str(len(np.unique(Test_Data["CustID"]))))
    st.write("Number of unique movies in trTestain data = ", str(len(np.unique(Test_Data["MovieID"]))))
    st.write("Highest value of a User ID = ", str(max(Test_Data["CustID"].values)))
    st.write("Highest value of a Movie ID =  ", str(max(Test_Data["MovieID"].values)))

def changingLabels(number):
    return str(number/10**6) + "M"

plt.figure(figsize = (12, 8))
ax = sns.countplot(x="Ratings", data=Train_Data)

ax.set_yticklabels([changingLabels(num) for num in ax.get_yticks()])

plt.tick_params(labelsize = 15)
plt.title("Distribution of Ratings in train data", fontsize = 20)
plt.xlabel("Ratings", fontsize = 20)
plt.ylabel("Number of Ratings(Millions)", fontsize = 20)
st.pyplot()
st.write("This graph will  show how **Distribution of Ratings** which shows the overall maturity level of the whole series and is provided by the audience :smile: ")

Train_Data["DayOfWeek"] = Train_Data.Date.dt.weekday_name
plt.figure(figsize = (10,8))
ax = Train_Data.resample("M", on = "Date")["Ratings"].count().plot()
ax.set_yticklabels([changingLabels(num) for num in ax.get_yticks()])
ax.set_title("Number of Ratings per Month", fontsize = 20)
ax.set_xlabel("Date", fontsize = 20)
ax.set_ylabel("Number of Ratings Per Month(Millions)", fontsize = 20)
plt.tick_params(labelsize = 15)
st.pyplot()
st.write("This Graph will represents the **Number of Ratings Per Month** means counts of ratings grouped by months :smile:")

st.write("**Analysis of Ratings given by user**")
no_of_rated_movies_per_user = Train_Data.groupby(by = "CustID")["Ratings"].count().sort_values(ascending = False)
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(14,7))
sns.kdeplot(no_of_rated_movies_per_user.values, shade = True, ax = axes[0])
axes[0].set_title("Fig1", fontsize = 18)
axes[0].set_xlabel("Number of Ratings by user", fontsize = 18)
axes[0].tick_params(labelsize = 15)
sns.kdeplot(no_of_rated_movies_per_user.values, shade = True, cumulative = True, ax = axes[1])
axes[1].set_title("Fig2", fontsize = 18)
axes[1].set_xlabel("Number of Ratings by user", fontsize = 18)
axes[1].tick_params(labelsize = 15)
fig.subplots_adjust(wspace=2)
plt.tight_layout()
st.pyplot()

st.write("Above fig1 graph shows that almost all of the users give very few ratings. There are very **few users who's ratings count is high** .Similarly, above fig2 graph shows that **almost 99% of users give very few ratings**")
quantiles = no_of_rated_movies_per_user.quantile(np.arange(0,1.01,0.01))
fig = plt.figure(figsize = (10, 6))
axes = fig.add_axes([0.1,0.1,1,1])
axes.set_title("Quantile values of Ratings Per User", fontsize = 20)
axes.set_xlabel("Quantiles", fontsize = 20)
axes.set_ylabel("Ratings Per User", fontsize = 20)
axes.plot(quantiles)
plt.scatter(x = quantiles.index[::5], y = quantiles.values[::5], c = "blue", s = 70, label="quantiles with 0.05 intervals")
plt.scatter(x = quantiles.index[::25], y = quantiles.values[::25], c = "red", s = 70, label="quantiles with 0.25 intervals")
plt.legend(loc='upper left', fontsize = 20)
for x, y in zip(quantiles.index[::25], quantiles.values[::25]):
    plt.annotate(s = '({},{})'.format(x, y), xy = (x, y), fontweight='bold', fontsize = 16, xytext=(x-0.05, y+180))
axes.tick_params(labelsize = 15)
st.pyplot()

st.write("this graph shows the Quantile values of Ratings Per User")
st.write("**Analysis of Ratings Per Movie** :smile:")
no_of_ratings_per_movie = Train_Data.groupby(by = "MovieID")["Ratings"].count().sort_values(ascending = False)
fig = plt.figure(figsize = (12, 6))
axes = fig.add_axes([0.1,0.1,1,1])
plt.title("Number of Ratings Per Movie", fontsize = 20)
plt.xlabel("Movie", fontsize = 20)
plt.ylabel("Count of Ratings", fontsize = 20)
plt.plot(no_of_ratings_per_movie.values)
plt.tick_params(labelsize = 15)
axes.set_xticklabels([])
st.pyplot()

st.write("This graph shows the number of rating(in count) each movie achieved by the audience, which clearly shows that there are some movies which are very popular and were rated by many users as comapared to other movies ")
st.write("**Analysis of Movie Ratings on Day of Week** :smile:")
fig = plt.figure(figsize = (12, 8))
axes = sns.countplot(x = "DayOfWeek", data = Train_Data)
axes.set_title("Day of week VS Number of Ratings", fontsize = 20)
axes.set_xlabel("Day of Week", fontsize = 20)
axes.set_ylabel("Number of Ratings", fontsize = 20)
axes.set_yticklabels([changingLabels(num) for num in ax.get_yticks()])
axes.tick_params(labelsize = 15)
st.pyplot()

st.write("This graph will show Analysis of Movie Ratings on Day of Week in bar graph format ,here clearly visible that on sturday & sunday users are least interested in providing ratings ")
fig = plt.figure(figsize = (12, 8))
axes = sns.boxplot(x = "DayOfWeek", y = "Ratings", data = Train_Data)
axes.set_title("Day of week VS Number of Ratings", fontsize = 20)
axes.set_xlabel("Day of Week", fontsize = 20)
axes.set_ylabel("Number of Ratings", fontsize = 20)
axes.tick_params(labelsize = 15)
st.pyplot()

st.write("This graph will show Analysis of Movie Ratings on Day of Week in box plot format ,here clearly visible that on sturday & sunday users are least interested in providing ratings ")
average_ratings_dayofweek = Train_Data.groupby(by = "DayOfWeek")["Ratings"].mean()
st.write("**Average Ratings on Day of Weeks**")
st.write(average_ratings_dayofweek)
st.write("**This Average Ratings on Day of Weeks will represented in graphical format** ")
st.area_chart(average_ratings_dayofweek)
st.write("this graph represents that average rating is mostly lies between 3 to 4.")
st.write("**Distribution of Movie ratings amoung Users**")
plt.scatter(Test_Data["CustID"],Test_Data["MovieID"])
st.pyplot()
