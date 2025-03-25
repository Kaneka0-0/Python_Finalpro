import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load and Prepare the Data

def read_gs():
    return pd.read_csv("GS_Music.csv")

df = read_gs()
print("Data loaded. Total rows:", len(df))

# Clean & Rename
df.replace([np.inf, -np.inf], np.nan, inplace=True)
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

# Filter for users aged 15-60 and Hip-Hop listeners on Spotify
filtered_df = df[(df['age'] >= 15) & (df['age'] <= 60) & (df['top_genre'] == 'Hip-Hop') & (df['platform'] == 'Spotify')]


# Feature Selection & Cleaning

features = ['age', 'songs_liked', 'engagement(%)', 'repeat(%)']
target = 'minutes_streamed'

data = filtered_df[features + [target]].dropna()
if data.empty:
    raise ValueError("Filtered data is empty. Check filters or missing values.")

X = data[features]
y = data[target]


# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Build a Pipeline with Random Forest

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])

# Hyperparameter Tuning
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)


# Evaluation

y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

n = len(y_test)
p = X_test.shape[1]
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("\nModel Performance:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.3f}")
print(f"Adjusted R2: {adjusted_r2:.3f}")


# Predict New User Input

new_user_input = np.array([[24, 120, 45, 60]])  # [age, songs_liked, engagement(%), repeat(%)]
predicted_minutes = best_model.predict(new_user_input)
print(f"\nPredicted Minutes Streamed for New User: {predicted_minutes[0]:.2f} minutes")


# Plot Actual vs Predicted

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
plt.xlabel("Actual Minutes Streamed")
plt.ylabel("Predicted Minutes Streamed")
plt.title("Actual vs Predicted (Random Forest)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'red')
plt.show()
