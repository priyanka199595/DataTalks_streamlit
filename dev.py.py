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
#matplotlib.use('TkAgg')
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


st.title('DataTalks')
st.write("Here **powerful and interactive graphics** will help you to precisely analyse :sunglasses:")

if not os.path.isfile("Data/NetflixRatings.csv"): 
#This line: "os.path.isfile("../Data/NetflixRatings.csv")" simply checks that is there a file with the name "NetflixRatings.csv" in the 
#in the folder "/Data/". If the file is present then it return true else false
    startTime = datetime.now()
    data = open("Data/NetflixRatings.csv", mode = "w") #this line simply creates the file with the name "NetflixRatings.csv" in 
    #write mode in the folder "Data".
#     files = ['../Data/combined_data_1.txt','../Data/combined_data_2.txt', '../Data/combined_data_3.txt', '../Data/combined_data_4.txt']
    files = ['Data/combined_data_2.txt']
    for file in files:
        print("Reading from file: "+str(file)+"...")
        with open(file) as f:  #you can think of this command "with open(file) as f" as similar to 'if' statement or a sort of 
            #loop statement. This command says that as long as this file is opened, perform the underneath operation.
            for line in f:
                line = line.strip() #line.strip() clears all the leading and trailing spaces from the string, as here each line
                #that we are reading from a file is a string.
                if line.endswith(":"):
                    movieID = line.replace(":", "") #this will remove the trailing semi-colon and return us the leading movie ID.
                else:
                    #here, in the below code we have first created an empty list with the name "row "so that we can insert movie ID 
                    #at the first position and rest customerID, rating and date in second position. After that we have separated all 
                    #four namely movieID, custID, rating and date with comma and converted a single string by joining them with comma.
                    #then finally written them to our output ".csv" file.
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
if not os.path.isfile("../Data/NetflixData.pkl"):
    startTime = datetime.now()
    Final_Data = pd.read_csv("../Data/NetflixRatings.csv", sep=",", names = ["MovieID","CustID", "Ratings", "Date"])
    Final_Data["Date"] = pd.to_datetime(Final_Data["Date"])
    Final_Data.sort_values(by = "Date", inplace = True)
    print("Time taken for execution of above code = "+str(datetime.now() - startTime))
    st.write("data frame created")
else:
    st.write("data frame already present")

# storing pandas dataframe as a picklefile for later use
if not os.path.isfile("../Data/NetflixData.pkl"):
    Final_Data.to_pickle("../Data/NetflixData.pkl")
    st.write("pkl created")
else:
    Final_Data = pd.read_pickle("../Data/NetflixData.pkl")
    st.write("pkl already present")

st.write(Final_Data)
st.area_chart(Final_Data.head())


Final_Data.describe()["Ratings"]

if not os.path.isfile("../Data/TrainData.pkl"):
    Final_Data.iloc[:int(Final_Data.shape[0]*0.80)].to_pickle("../Data/TrainData.pkl")
    Train_Data = pd.read_pickle("../Data/TrainData.pkl")
    Train_Data.reset_index(drop = True, inplace = True)
else:
    Train_Data = pd.read_pickle("../Data/TrainData.pkl")
    Train_Data.reset_index(drop = True, inplace = True)

if not os.path.isfile("../Data/TestData.pkl"):
    Final_Data.iloc[int(Final_Data.shape[0]*0.80):].to_pickle("../Data/TestData.pkl")
    Test_Data = pd.read_pickle("../Data/TestData.pkl")
    Test_Data.reset_index(drop = True, inplace = True)
else:
    Test_Data = pd.read_pickle("../Data/TestData.pkl")
    Test_Data.reset_index(drop = True, inplace = True)


st.write(Train_Data)
st.area_chart(Train_Data.head())


st.write(Test_Data)
st.area_chart(Test_Data.head())


def changingLabels(number):
    return str(number/10**6) + "M"

plt.figure(figsize = (8,6))
ax = Train_Data.resample("M", on = "Date")["Ratings"].count().plot()
ax.set_yticklabels([changingLabels(num) for num in ax.get_yticks()])
ax.set_title("Number of Ratings per month", fontsize = 20)
ax.set_xlabel("Date", fontsize = 20)
ax.set_ylabel("Number of Ratings Per Month(Millions)", fontsize = 20)
plt.tick_params(labelsize = 15)
plt.show()
