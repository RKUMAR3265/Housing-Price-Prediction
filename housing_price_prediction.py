# ---------------------------------------
# TASK 2: PREDICTING HOUSING PRICES
# ---------------------------------------

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Step 2: Load dataset
data = pd.read_csv(r"D:\DATA ANALYSIS\boston-housing-dataset.csv")  # Update path if needed
print("✅ Dataset Loaded Successfully!\n")
print(data.head())

# Step 3: Basic info
print("\nDataset Info:")
print(data.info())

# Step 4: Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# Step 5: Explore data (optional visualization)
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Step 6: Define features (X) and target (y)
X = data.drop(['MEDV'], axis=1)   # MEDV = Median value of owner-occupied homes
y = data['MEDV']

# Step 7: Split data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Normalize/scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Train Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 10: Make predictions
y_pred = model.predict(X_test_scaled)

# Step 11: Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Step 12: Plot predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Step 13: Show coefficients (feature importance)
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

print("\n🔍 Feature Importance:")
print(coefficients)
