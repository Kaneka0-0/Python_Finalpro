# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from sklearn.preprocessing import PolynomialFeatures
# from read_gs import read_gs  # Import the function from read_gs.py

# # Load the data
# # df = read_gs()

# # # Replace infinite values with NaN
# # df.replace([np.inf, -np.inf], np.nan, inplace=True)
# df = pd.read_csv("GS_Music.csv")


# df = read_gs()
# print("Length of the data:", len(df))
# print ("Column Heads\n",df.head())

# # Rename columns if necessary
# df.rename(columns={
#     "User_ID": "user_id",
#     "Age": "age",
#     "Country": "country",
#     "Streaming Platform": "platform",
#     "Top Genre": "top_genre",
#     "Minutes Streamed Per Day": "minutes_streamed",
#     "Number of Songs Liked": "songs_liked",   
#     "Most Played Artist": "top_artist",
#     "Subscription Type": "subscription",
#     "Listening Time (Morning/Afternoon/Night)": "listening_time",
#     "Discover Weekly Engagement (%)": "engagement(%)",
#     "Repeat Song Rate (%)": "repeat(%)"
# }, inplace=True)

# print("Column Heads (rename)\n",df.head())

# plt.figure(figsize=(10, 6))
# sns.countplot(data=df, x='platform')
# plt.title('Count of Users by Streaming Platform')
# plt.xlabel('Streaming Platform')
# plt.ylabel('Count')
# plt.show()

# # Filter the data for age 15 to 60 and platform "Spotify"
# filtered_df = df[(df['age'] >= 15) & (df['age'] <= 60) & (df['platform'] == 'Spotify')]

# # Identify the top artist among these users
# top_artist = filtered_df['top_artist'].mode()[0]

# # Filter the data for the top artist
# top_artist_df = filtered_df[filtered_df['top_artist'] == top_artist]

# # Regression plot for the repeat rate of the top artist by age
# plt.figure(figsize=(10, 6))
# sns.regplot(data=top_artist_df, x='age', y='repeat(%)')
# plt.title(f'Repeat Rate for Top Artist ({top_artist}) on Spotify (Users Aged 15-60)')
# plt.xlabel('Age')
# plt.ylabel('Repeat Rate (%)')
# plt.grid(True)
# plt.show()

# # Select only numeric columns (int and float) for the correlation matrix
# numeric_df = df.select_dtypes(include=[np.number])

# # Create a correlation matrix from the numeric DataFrame to use in heatmap
# corr_matrix = numeric_df.corr()

# # Plot the heatmap- put details in the heatmap- only take numbers
# plt.figure(figsize=(6, 4))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation Heatmap')
# plt.show()

# # Filter the data for platform "Spotify"
# spotify_df = df[df['platform'] == 'Spotify']

# # Calculate the mean Discover Weekly Engagement by Subscription Type for Spotify
# mean_engagement_by_subscription = spotify_df.groupby(['subscription', 'age'])['engagement(%)'].mean().reset_index()

# # Plot the mean Discover Weekly Engagement by Subscription Type for Spotify
# plt.figure(figsize=(8, 6))
# sns.countplot(data=mean_engagement_by_subscription, y='subscription', x='engagement(%)', hue='age')
# plt.title('Mean Discover Weekly Engagement by Subscription Type and age (Spotify)')
# plt.ylabel('Subscription Type')
# plt.xlabel('Mean Discover Weekly Engagement (%)')
# plt.legend(title='Age')
# plt.show()

# # Predict results
# # Defining features and target
# features = ['age', 'minutes_streamed', 'engagement(%)', 'repeat(%)']
# target = 'songs_liked'

# # Prepare the data (drop NaN)
# data = filtered_df[features + [target]].dropna()
# X = data[features]
# y = data[target]

# # Split into train and test sets (70% train, 30% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Train a linear regression model
# model = LinearRegression()  # Model Creation- Creating an instance of the Linear Regression model
# model.fit(X_train, y_train)  # Fitting the model to the training data

# # Making predictions on the test data
# y_pred = model.predict(X_test)

# # Calculate the metrics
# mse = mean_squared_error(y_test, y_pred)  # Calculating Mean Squared Error
# r2 = r2_score(y_test, y_pred)  # Calculating R-squared score
# mae = mean_absolute_error(y_test, y_pred)  # Calculating Mean Absolute Error
# rmse = np.sqrt(mse)  # Calculating Root Mean Squared Error

# print(f"Mean Absolute Error (MAE): \n {mae}\n ")
# print(f"Mean Squared Error (MSE): \n {mse}\n ")  # Printing the Mean Squared Error
# print(f"Mean Absolute Error (MAE): \n {mae}\n ")  # Printing the Mean Absolute Error
# print(f"Root Mean Squared Error (RMSE): \n {rmse}\n ")  # Printing the Root Mean Squared Error
# print(f"R-Squared Score (R2): \n {r2}\n ")  # Printing the R-squared score



# # top_genre distribution plot
# plt.figure(figsize=(8, 5))
# sns.histplot(df['top_genre'], bins=20, kde=True, color='blue')
# plt.title('Distribution of Listener top_genre')
# plt.xlabel('top_genre')
# plt.ylabel('Count')
# plt.show()

# # Subscription Type vs. streaming minutes
# plt.figure(figsize=(12, 8))
# sns.countplot(data=df, x='subscription', hue='platform')
# plt.title('Subscription Type vs. Streaming Platform')
# plt.xlabel('Subscription Type')
# plt.ylabel('Count')
# plt.show()
# # plt.title('Subscription vs. songs_liked')
# # plt.xlabel('Subscription')
# # plt.ylabel('songs_liked')
# # plt.show()

# # # Residual Analysis
# # residuals = y_test - y_pred  # Calculating residuals
# # plt.figure(figsize=(8, 5))  # Setting the figure size for the residual plot
# # sns.histplot(residuals, bins=30, kde=True)  # Creating a histogram of the residuals
# # plt.xlabel("Residuals")  # Labeling the x-axis
# # plt.ylabel("Frequency")  # Labeling the y-axis
# # plt.title("Residual Distribution")  # Setting the title of the plot
# # plt.show()  # Displaying the plot

# # Minutes Streamed vs. engagement(%)-----------------------new code
# plt.figure(figsize=(8, 5))
# sns.scatterplot(x=df['songs_liked'], y=df['top_genre'], alpha=0.5, color='green')
# plt.title('songs_liked vs. top_genre')
# plt.xlabel('songs_liked')
# plt.ylabel('top_genre')
# plt.show()

# # Adjusted R-Squared Calculation
# n = len(y_test)  # Number of observations
# p = X_test.shape[1]  # Number of predictors
# adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))  # Calculating Adjusted R-squared
# print(f"Adjusted R-Squared: \n {adjusted_r2}")  # Printing the Adjusted R-squared

# # Predicting new user input [age, songs_liked, engagement(%), repeat(%)]

# new_user = np.array([[24, 120, 45, 60]])
# predicted_minutes = model.predict(new_user)
# print(f"Predicted Minutes Streamed Per Day: {predicted_minutes[0]}")

# ###### Additional code for the model

# # Polynomial regression
# poly = PolynomialFeatures(degree=2)
# X_poly = poly.fit_transform(X_train)
# model_poly = LinearRegression()
# model_poly.fit(X_poly, y_train)
# y_poly_pred = model_poly.predict(poly.transform(X_test))


# # Evaluate polynomial regression
# print("Polynomial R²:\n", r2_score(y_test, y_poly_pred)) 


# from sklearn.ensemble import RandomForestRegressor

# # rf_model = RandomForestRegressor(random_state=42)
# # rf_model.fit(X_train, y_train)
# # y_rf_pred = rf_model.predict(X_test)

# # # Evaluate Visualize predicted vs. actual values
# # print("Random Forest R²:\n", r2_score(y_test, y_rf_pred))


# plt.figure(figsize=(8, 5))
# plt.scatter(y_test, y_pred, alpha=0.7)
# plt.xlabel("Actual songs_liked")
# plt.ylabel("Predicted songs_liked")
# plt.title("Actual vs. Predicted Song liked (Linear Regression)")
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'red')
# plt.show()


# # # Add feature importance (if using tree-based models)
# # importances = rf_model.feature_importances_
# # feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
# # feature_importance.sort_values('Importance', ascending=True, inplace=True)
# # print(feature_importance)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
df = pd.read_csv("GS_Music.csv")

# Rename columns
column_mapping = {
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
}
df.rename(columns=column_mapping, inplace=True)

# Display first rows
print("Dataset Sample:")
print(df.head())

# Visualizing user count per streaming platform
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='platform')
plt.title('User Count per Streaming Platform')
plt.xlabel('Streaming Platform')
plt.ylabel('Count')
plt.show()

# Filter dataset for Spotify users aged 15-60
filtered_df = df[(df['age'] >= 15) & (df['age'] <= 60) & (df['platform'] == 'Spotify')]

# Identify most played artist
top_artist = filtered_df['top_artist'].mode()[0]
top_artist_df = filtered_df[filtered_df['top_artist'] == top_artist]

# Regression plot: Repeat Rate vs Age for the most played artist
plt.figure(figsize=(10, 6))
sns.regplot(data=top_artist_df, x='age', y='repeat(%)')
plt.title(f'Repeat Rate for {top_artist} on Spotify (Aged 15-60)')
plt.xlabel('Age')
plt.ylabel('Repeat Rate (%)')
plt.grid(True)
plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# Calculate mean engagement by subscription type and age
spotify_df = df[df['platform'] == 'Spotify']
mean_engagement = spotify_df.groupby(['subscription', 'age'])['engagement(%)'].mean().reset_index()

# Bar plot: Engagement by subscription type and age
plt.figure(figsize=(8, 6))
sns.barplot(data=mean_engagement, x='subscription', y='engagement(%)', hue='age')
plt.title('Mean Discover Weekly Engagement by Subscription Type and Age (Spotify)')
plt.ylabel('Mean Engagement (%)')
plt.xlabel('Subscription Type')
plt.legend(title='Age')
plt.show()

# Linear Regression: Predicting number of songs liked
features = ['age', 'minutes_streamed', 'engagement(%)', 'repeat(%)']
target = 'songs_liked'
data = filtered_df[features + [target]].dropna()
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
n = len(y_test)
p = X_test.shape[1]
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print(f"MAE: {mae}\nMSE: {mse}\nRMSE: {rmse}\nR-Squared: {r2}\nAdjusted R-Squared: {adjusted_r2}")

# Predict for a new user input
new_user = np.array([[24, 120, 45, 60]])
predicted_songs_liked = model.predict(new_user)
print(f"Predicted Songs Liked: {predicted_songs_liked[0]}")

# Actual vs. Predicted scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Songs Liked")
plt.ylabel("Predicted Songs Liked")
plt.title("Actual vs. Predicted Songs Liked (Linear Regression)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'red')
plt.show()


