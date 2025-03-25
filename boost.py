import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("GS_Music.csv")

# Clean column names
df.rename(columns={
    "Age": "age",
    "Number of Songs Liked": "songs_liked",
    "Minutes Streamed Per Day": "minutes_streamed",
    "Discover Weekly Engagement (%)": "engagement",
    "Repeat Song Rate (%)": "repeat"
}, inplace=True)

# Drop rows with any missing values in the required columns
df = df[['age', 'engagement', 'repeat', 'minutes_streamed', 'songs_liked']].dropna()

# Remove outliers in the target variable (songs_liked)
Q1 = df['songs_liked'].quantile(0.25)
Q3 = df['songs_liked'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['songs_liked'] >= Q1 - 1.5 * IQR) & (df['songs_liked'] <= Q3 + 1.5 * IQR)]

# Define features and target
features = ['age', 'engagement', 'repeat', 'minutes_streamed']
target = 'songs_liked'

X = df[features]
y = df[target]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning with GridSearchCV for XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search_xgb = GridSearchCV(XGBRegressor(random_state=42), param_grid_xgb, cv=5, scoring='neg_mean_squared_error')
grid_search_xgb.fit(X_train, y_train)

# Best XGBoost model
best_model_xgb = grid_search_xgb.best_estimator_

# Predict with XGBoost
y_pred_xgb = best_model_xgb.predict(X_test)

# Evaluation for XGBoost
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
n = len(y_test)
p = X_test.shape[1]
adjusted_r2_xgb = 1 - ((1 - r2_xgb) * (n - 1) / (n - p - 1))

# Hyperparameter tuning with GridSearchCV for RandomForest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
}

grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)

# Best RandomForest model
best_model_rf = grid_search_rf.best_estimator_

# Predict with RandomForest
y_pred_rf = best_model_rf.predict(X_test)

# Evaluation for RandomForest
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
adjusted_r2_rf = 1 - ((1 - r2_rf) * (n - 1) / (n - p - 1))

# Show results
print("XGBoost Results:")
print(f"Best Parameters: {grid_search_xgb.best_params_}")
print(f"MAE: {mae_xgb:.2f}")
print(f"RMSE: {rmse_xgb:.2f}")
print(f"R2: {r2_xgb:.3f}")
print(f"Adjusted R2: {adjusted_r2_xgb:.3f}")

print("\nRandomForest Results:")
print(f"Best Parameters: {grid_search_rf.best_params_}")
print(f"MAE: {mae_rf:.2f}")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R2: {r2_rf:.3f}")
print(f"Adjusted R2: {adjusted_r2_rf:.3f}")
