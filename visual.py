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


# Filter the data for age 15 to 45 and platform "Spotify"
filtered_df = df[(df['age'] >= 15) & (df['age'] <= 60) & (df['platform'] == 'Spotify')]

# Identify the top artist among these users
top_artist = filtered_df['top_artist'].mode()[0]

# Filter the data for the top artist
top_artist_df = filtered_df[filtered_df['top_artist'] == top_artist]

# Regression plot for the repeat rate of the top artist by age
plt.figure(figsize=(10, 6))
sns.regplot(data=top_artist_df, x='age', y='repeat(%)')
plt.title(f'Repeat Rate for Top Artist ({top_artist}) on Spotify (Users Aged 15-60)')
plt.xlabel('Age')
plt.ylabel('Repeat Rate (%)')
plt.grid(True)
plt.show()


# Select only numeric columns (int and float) for the correlation matrix
numeric_df = df.select_dtypes(include=[np.number])

# Create a correlation matrix from the numeric DataFrame to use in heatmap
corr_matrix = numeric_df.corr()

# Plot the heatmap- put details in the heatmap- only take numbers
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Filter the data for platform "Spotify"
spotify_df = df[df['platform'] == 'Spotify']

# Calculate the mean Discover Weekly Engagement by Subscription Type for Spotify
mean_engagement_by_subscription = spotify_df.groupby(['subscription', 'age'])['engagement(%)'].mean().reset_index()

# Plot the mean Discover Weekly Engagement by Subscription Type for Spotify
plt.figure(figsize=(8, 6))
sns.barplot(data=mean_engagement_by_subscription, y='subscription', x='engagement(%)', hue='age')
plt.title('Mean Discover Weekly Engagement by Subscription Type and age (Spotify)')
plt.ylabel('Subscription Type')
plt.xlabel('Mean Discover Weekly Engagement (%)')
plt.legend(title='Age')
plt.show()


#predict results
features = ['age', 'songs_liked', 'engagement(%)', 'repeat(%)']
target = 'minutes_streamed'
data = filtered_df[features + [target]].dropna()
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-Squared Score (R2): {r2}")

# Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.show()

# Adjusted R-Squared Calculation
n = len(y_test)  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
print(f"Adjusted R-Squared: {adjusted_r2}")