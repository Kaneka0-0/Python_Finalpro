# read the dataset and return the data and the target- 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = "GS_Music.csv"
df = pd.read_csv(file_path)
print("Length of the data:", len(df))

print (df.info())
print (df.describe())
print (df.head())
