import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns   
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import os 

# Suppress the Tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Load the dataset
data = pd.read_csv('Data\\MicrosoftStock.csv')
# print(data.head())  
# print(data.info())
# print(data.describe())

#Initial Data Visiualization to get better undersatnding of the data

# plot 1 : Open and Close Stock Prices over time (Checking for any trends or patterns or how close the stock prices are to each other)
# plt.figure(figsize = (8, 6))
# plt.plot(data["date"] , data["open"] , label = "Open Stock" , color = "blue")
# plt.plot(data["date"] , data["close"] , label = "Close Stock" , color = "red")
# plt.title("Microsoft Stock Prices")
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.legend()
# plt.show()

# plot 2 : Volume of Stocks traded over time(checking for any anomalies or outliers)
# plt.figure(figsize = (8, 6))
# plt.plot(data["date"] , data["volume"] , label= "Volume of Stocks Traded" , color = "green")
# plt.title("Volume of Stocks Traded Over Time")
# plt.xlabel("Date")
# plt.ylabel("Volume")
# plt.legend()
# plt.show()

# we want to see how correlated the features are to each other
# for this we will draw the corelation matrix
# corelation matrix can only be drawn on numerical columns 
# extracting the numerical column sfrom the dataset we have
 

numerical_columns = data.select_dtypes(include = ["float64" , "int64"])
#correlation metrix to check the corelation between the numerical features

# Plot 2 : Check for Corelation between the features
plt.figure(figsize =(8 , 6))
sns.heatmap(numerical_columns.corr() , annot = True , cmap= "coolawrm")





