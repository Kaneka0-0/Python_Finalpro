import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from read_gs import read_gs  # Import the function from read_gs.py

# Load the data
df = read_gs()

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

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

# Filter the data for age 15 to 60 and platform "Spotify"
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

# Predict results
# Defining features and target
features = ['age', 'songs_liked', 'engagement(%)', 'repeat(%)']
target = 'minutes_streamed'

# Prepare the data (drop NaN)
data = filtered_df[features + [target]].dropna()
X = data[features]
y = data[target]

# Split into train and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a linear regression model
model = LinearRegression()  # Model Creation- Creating an instance of the Linear Regression model
model.fit(X_train, y_train)  # Fitting the model to the training data

# Making predictions on the test data
y_pred = model.predict(X_test)

# Calculate the metrics
mse = mean_squared_error(y_test, y_pred)  # Calculating Mean Squared Error
r2 = r2_score(y_test, y_pred)  # Calculating R-squared score
mae = mean_absolute_error(y_test, y_pred)  # Calculating Mean Absolute Error
rmse = np.sqrt(mse)  # Calculating Root Mean Squared Error

print(f"Mean Absolute Error (MAE): \n {mae}\n ")
print(f"Mean Squared Error (MSE): \n {mse}\n ")  # Printing the Mean Squared Error
print(f"Mean Absolute Error (MAE): \n {mae}\n ")  # Printing the Mean Absolute Error
print(f"Root Mean Squared Error (RMSE): \n {rmse}\n ")  # Printing the Root Mean Squared Error
print(f"R-Squared Score (R2): \n {r2}\n ")  # Printing the R-squared score



# top_genre distribution plot
plt.figure(figsize=(8, 5))
sns.histplot(df['top_genre'], bins=20, kde=True, color='blue')
plt.title('Distribution of Listener top_genre')
plt.xlabel('top_genre')
plt.ylabel('Count')
plt.show()

# Subscription Type vs. streaming minutes
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='subscription', y='minutes_streamed', palette='colorblind')
plt.title('Subscription vs. minutes_streamed')
plt.xlabel('Subscription')
plt.ylabel('minutes_streamed')
plt.show()

# # Residual Analysis
# residuals = y_test - y_pred  # Calculating residuals
# plt.figure(figsize=(8, 5))  # Setting the figure size for the residual plot
# sns.histplot(residuals, bins=30, kde=True)  # Creating a histogram of the residuals
# plt.xlabel("Residuals")  # Labeling the x-axis
# plt.ylabel("Frequency")  # Labeling the y-axis
# plt.title("Residual Distribution")  # Setting the title of the plot
# plt.show()  # Displaying the plot

# Minutes Streamed vs. Repeat Song Rate-----------------------new code
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['minutes_streamed'], y=df['engagement(%)'], alpha=0.5, color='green')
plt.title('Minutes Streamed vs. engagement(%)')
plt.xlabel('Minutes Streamed Per Day')
plt.ylabel('engagement(%)')
plt.show()

# Adjusted R-Squared Calculation
n = len(y_test)  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))  # Calculating Adjusted R-squared
print(f"Adjusted R-Squared: \n {adjusted_r2}")  # Printing the Adjusted R-squared

# Predicting new user input [age, songs_liked, engagement(%), repeat(%)]

new_user = np.array([[24, 120, 45, 60]])
predicted_minutes = model.predict(new_user)
print(f"Predicted Minutes Streamed Per Day: {predicted_minutes[0]}")

###### Additional code for the model

# Polynomial regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
model_poly = LinearRegression()
model_poly.fit(X_poly, y_train)
y_poly_pred = model_poly.predict(poly.transform(X_test))


# Evaluate polynomial regression
print("Polynomial R²:\n", r2_score(y_test, y_poly_pred)) 


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_rf_pred = rf_model.predict(X_test)

# Evaluate Visualize predicted vs. actual values
print("Random Forest R²:\n", r2_score(y_test, y_rf_pred))


plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Minutes Streamed")
plt.ylabel("Predicted Minutes Streamed")
plt.title("Actual vs. Predicted Minutes Streamed (Linear Regression)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'red')
plt.show()


# Add feature importance (if using tree-based models)
importances = rf_model.feature_importances_
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance.sort_values('Importance', ascending=True, inplace=True)
print(feature_importance)

