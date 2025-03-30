import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance

# Load the data
df = pd.read_csv("GS_Music.csv")

# Rename columns
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

# Data Cleaning and Preprocessing
def preprocess_data(df):
    # Remove duplicates
    df.drop_duplicates(subset=['user_id'], inplace=True)
    
    # Handle missing values
    df.dropna(inplace=True)
    
    # Remove outliers using IQR method for numeric columns
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # Columns to check for outliers
    numeric_cols = ['age', 'minutes_streamed', 'songs_liked', 'engagement(%)', 'repeat(%)']
    for col in numeric_cols:
        df = remove_outliers(df, col)
    
    return df

# Feature Engineering
def feature_engineering(df):
    # Age groups
    df['age_group'] = pd.cut(df['age'], 
        bins=[0, 18, 25, 35, 45, 55, 100], 
        labels=['Under 18', '18-25', '26-35', '36-45', '46-55', '55+'])
    
    # Streaming intensity
    df['streaming_intensity'] = df['minutes_streamed'] / df['songs_liked']
    
    # Engagement score
    df['engagement_score'] = df['engagement(%)'] * df['repeat(%)'] / 100
    
    return df

# Preprocessing and Feature Engineering
df_cleaned = preprocess_data(df)
df_featured = feature_engineering(df_cleaned)

# Convert categorical(String) features to numeric counts for correlation analysis
def convert_categorical_to_counts(df, categorical_features):
    for col in categorical_features:
        # Replace each category with its count in the dataset
        df[col + '_count'] = df[col].map(df[col].value_counts())
    return df

# Prepare features and target
categorical_features = ['platform', 'top_genre', 'subscription', 'listening_time', 'age_group', 'country']
numeric_features = ['age', 'minutes_streamed', 'engagement(%)', 'repeat(%)', 
                    'streaming_intensity', 'engagement_score']
target = 'songs_liked'

# Apply the conversion for correlation heatmap
df_transformed = convert_categorical_to_counts(df_featured, categorical_features)

# Combine numeric features and transformed categorical features
all_features = numeric_features + [col + '_count' for col in categorical_features]
df_combined = df_transformed[all_features]

# Create a correlation matrix
correlation_matrix = df_combined.corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap (Numeric + Transformed Categorical Features)')
plt.tight_layout()
plt.show()

# Preprocessing for numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create two different models for comparison
ridge_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Prepare the data
X = df_featured[numeric_features + categorical_features]
y = df_featured[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    print("Model Performance Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-Squared Score (R²): {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    
    return y_pred

# Train and evaluate Random Forest
print("\nRandom Forest Regression Results:")
rf_pipeline.fit(X_train, y_train)
rf_pred = evaluate_model(rf_pipeline, X_test, y_test)

# Cross-validation
cv_scores_ridge = cross_val_score(ridge_pipeline, X, y, cv=5, scoring='r2')
cv_scores_rf = cross_val_score(rf_pipeline, X, y, cv=5, scoring='r2')

print("\nCross-Validation Scores:")
print(f"Ridge Regression - Mean CV R² Score: {cv_scores_ridge.mean():.4f} (+/- {cv_scores_ridge.std() * 2:.4f})")
print(f"Random Forest - Mean CV R² Score: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")

# Prediction Function
def predict_songs_liked(model, new_user_data):
    prediction = model.predict(new_user_data)
    return prediction[0]

# Visualization: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Songs Liked (Random Forest)')
plt.xlabel('Actual Songs Liked')
plt.ylabel('Predicted Songs Liked')
plt.show()

# Get feature names after preprocessing
feature_names = (
    numeric_features + 
    list(rf_pipeline.named_steps['preprocessor']
         .named_transformers_['cat']
         .get_feature_names_out(categorical_features))
)

# Get the trained random forest model
rf_model = rf_pipeline.named_steps['regressor']

# Calculate feature importances
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df = importance_df.sort_values('importance', ascending=False)

# Get top features (e.g., top 15) based on importance
top_n = 15
top_features = importance_df.head(top_n)['feature'].tolist()

# Create a feature importance heatmap
plt.figure(figsize=(14, 10))

# Reshape data for heatmap
importance_heatmap_data = importance_df.head(top_n).copy()
importance_heatmap_data['importance_normalized'] = importance_heatmap_data['importance'] / importance_heatmap_data['importance'].max()

# # Create a DataFrame with one row for the heatmap
# heatmap_df = pd.DataFrame(
#     [importance_heatmap_data['importance_normalized'].values],
#     columns=importance_heatmap_data['feature'].values
# )

# # Plot horizontal heatmap
# sns.heatmap(
#     heatmap_df, 
#     annot=importance_heatmap_data[['importance']].values,
#     fmt='.3f',
#     cmap='viridis',
#     cbar_kws={'label': 'Normalized Importance'}
# )

# plt.title('Random Forest Feature Importance Heatmap', fontsize=16)
# plt.xlabel('Features')
# plt.ylabel('Importance')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()