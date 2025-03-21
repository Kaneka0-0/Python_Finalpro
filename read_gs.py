# read the dataset and return the data and the target- 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
def read_gs():
    df = pd.read_csv("GS_Music.csv")
    return df

df = read_gs()
print("Length of the data:", len(df))

# Display the first few rows of the dataset
# print ("Column Heads\n",df.head()) - use df.head() to display the first few rows of the dataset- we command this out becasue we want to display the renamed columns.

#summary of the dataset
print("Suammary Statistic (Numeric values): \n",df.describe()) #summary of the dataset- use df.describe() to get the summary of the dataset.

#show data types
print("Data types: \n",df.dtypes)

#rename columns
df.rename(columns={
    "User_ID": "user_id",
    "Age": "age",
    "Country": "country",
    "Streaming Platform": "platform",
    "Top Genre": "top_genre",
    "Minutes Streamed Per Day": "minutes_streamed",
    "Number of Songs Liked": "songs_liked",   
    "Most Played Artist": "top_artist",
    "Subscription Type": "subscription",
    "Listening Time (Morning/Afternoon/Night)": "listening_time",
    "Discover Weekly Engagement (%)": "engagement(%)",
    "Repeat Song Rate (%)": "repeat(%)"
}, inplace=True)

#print("Column Heads (rename)\n",df.head())

#check for missing value
print("Missing values:\n",df.isnull().sum())



