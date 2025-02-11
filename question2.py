import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
try:
    df = pd.read_csv('weight-height.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'weight-height.csv' not found. Please check the file path.")
    exit()

# Display first few rows
print(df.head())

# Step 2: Scatter plot (Height vs. Weight)
plt.figure(figsize=(8, 5))
plt.scatter(df['Height'], df['Weight'], alpha=0.5, color='blue')
plt.title('Height vs Weight')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.show()

# Step 3: Choose regression model (Linear Regression)
X = df[['Height']]  # Independent variable
y = df['Weight']    # Dependent variable

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Step 4: Make predictions
y_pred = model.predict(X)

# Step 5: Plot actual vs predicted values
plt.figure(figsize=(8, 5))
plt.scatter(df['Height'], df['Weight'], alpha=0.5, label='Actual Data', color='blue')
plt.plot(df['Height'], y_pred, color='red', label='Linear Regression Line')
plt.title('Height vs Weight with Linear Regression')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.legend()
plt.show()

# Step 6: Compute RMSE and R²
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Step 7: Assessment
if r2 > 0.9:
    print("The model fits the data very well.")
elif 0.7 < r2 <= 0.9:
    print("The model fits the data well but has some variance.")
else:
    print("The model does not fit the data well; consider a non-linear model.")
