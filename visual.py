import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from read_gs import read_gs  # Import the function from read_gs.py

# Load the data
df = read_gs()

# Rename columns if necessary
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

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='platform')
plt.title('Count of Users by Streaming Platform')
plt.xlabel('Streaming Platform')
plt.ylabel('Count')
plt.show()


# Filter the data for age 15 to 45 and platform "YouTube"
filtered_df = df[(df['age'] >= 15) & (df['age'] <= 45) & (df['platform'] == 'YouTube')]

# Identify the top artist among these users
top_artist = filtered_df['top_artist'].mode()[0]

# Filter the data for the top artist
top_artist_df = filtered_df[filtered_df['top_artist'] == top_artist]

# Example of a regression plot
plt.figure(figsize=(10, 6))
sns.regplot(data=top_artist_df, x='age', y='repeat(%)')
plt.title(f'Repeat Rate for Top Artist ({top_artist}) on YouTube (Users Aged 15-45)')
plt.xlabel('Age')
plt.ylabel('Repeat Rate (%)')
plt.grid(True)
plt.show()


# Select only numeric columns (int and float) for the correlation matrix
numeric_df = df.select_dtypes(include=[np.number])

# Create a correlation matrix from the numeric DataFrame
corr_matrix = numeric_df.corr()

# Plot the heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Calculate the median Discover Weekly Engagement by Subscription Type
median_engagement_by_subscription = df.groupby('subscription')['engagement(%)'].median().reset_index()

# Plot the median Discover Weekly Engagement by Subscription Type
plt.figure(figsize=(10, 6))
sns.barplot(data=median_engagement_by_subscription, x='subscription', y='engagement(%)')
plt.title('Median Discover Weekly Engagement by Subscription Type')
plt.xlabel('Subscription Type')
plt.ylabel('Median Discover Weekly Engagement (%)')
plt.show()
