[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=18760428&assignment_repo_type=AssignmentRepo)

# Python Final Project

This project analyzes music streaming data and visualizes various insights using Python libraries such as pandas, seaborn, and matplotlib.This project analyzes music streaming data and visualizes various insights using Python libraries such as pandas, seaborn, and matplotlib.This project analyzes music streaming data and visualizes various insights using Python libraries such as pandas, seaborn, and matplotlib.

## Installationionion

First, download the required libraries. To install them, run the following command:oad the required libraries. To install them, run the following command:First, download the required libraries. To install them, run the following command:
```sh
pip install -r requirements.txt
```

to check the installed libraries and their versions, use:
' pip list '

Prediction: for our project we chose to predict 'minutes_streamed' on spotify. to see how much it will grown in the future.

Find accuracy speed/rate. (r2 or sth in the sense)

### Overview: 

This project aims to predict the number of minutes a user streams per day on Spotify based on various features like age, songs liked, engagement, and repeat song rate. The model leverages different regression techniques to assess the accuracy and improve the predictive performance.

Steps
1. Data Preparation:

- The dataset is loaded from a CSV file (GS_Music.csv).
- Column names are standardized to ensure consistency.
- Missing values are checked.
- Data is filtered to focus on users between the ages of 15-60 on the Spotify platform.

2. Data Exploration:

- A count plot of users based on streaming platforms.
- Regression plot showing the repeat rate for the top artist by age.
- A heatmap of the correlation matrix to visualize relationships between numeric features.

3. Feature Engineering and Data Splitting:

- Selects the relevant features (age, songs_liked, engagement(%), repeat(%)) and the target variable (minutes_streamed).
- Splits the data into training and testing sets (70% train, 30% test).

4. Modeling:

- A Linear Regression model is trained to predict the target variable (minutes_streamed).
- Evaluation metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared are calculated.

5. Residual Analysis:

- Residuals are analyzed by plotting their distribution to ensure that the model’s assumptions are valid.

6. Advanced Models:

- Polynomial Regression: A polynomial regression model is trained to handle non-linear relationships in the data.
- Random Forest Regressor: A random forest model is employed to capture complex interactions between features and provide feature importance insights.

7. Visualizations:

- A comparison of predicted vs. actual values for both linear regression and random forest models is plotted.
- Feature importance for the random forest model is also visualized.

Conclusion:

This code effectively demonstrates how to predict user behavior (in terms of minutes streamed per day) on Spotify using various regression models. Polynomial regression and RandomForestRegressor are employed to improve predictive accuracy by capturing non-linear relationships and complex feature interactions in the data.

### Why we need **Polynomial Regression**:

1. **Straight lines don't always fit well**: In regular linear regression, we fit a straight line to the data. But sometimes, the data follows a curve, not a straight line.
2. **Curves in data**: Polynomial regression helps when the relationship between the variables is curved, not just straight. It lets us draw a curved line instead of a straight one.
3. **Better fit**: By adding more terms (like x², x³), we can create a better-fitting line that captures the bending pattern in the data.

### Why we need **RandomForestRegressor**:
1. **Handling complex data**: Sometimes, the data is too messy or complex for a simple straight line (or even a curved line). In these cases, Random Forest can help.
2. **Multiple helpers (trees)**: A Random Forest works by asking a bunch of "helpers" (decision trees) to make predictions. Each helper looks at the data differently, so it can capture different patterns.
3. **Combining results**: After all the helpers make their predictions, Random Forest averages their answers to get a final, more accurate prediction. This is better than relying on just one helper (like a single decision tree).
4. **Handling non-linear relationships**: Random Forest can handle both linear and non-linear relationships, so it's great for more complicated data that doesn’t fit a simple line or curve.

In short:
- **Polynomial Regression** helps when data curves.
- **RandomForestRegressor** helps when data is too complex for a single line or curve.