# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load the sales dataset
file_path = 'train.csv'
data = pd.read_csv(file_path)
# Explore the dataset
print("Data shape:", data.shape)
print("\nData info:")
print(data.info())
print("\nData sample:")
print(data.head())

# Data preprocessing and visualie the sales trend over time
data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d/%m/%Y')

sales_by_date = data.groupby('Order Date')['Sales'].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(sales_by_date['Order Date'], sales_by_date['Sales'], label='Sales', color='red')
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Feature engineering - creating lagged features for capturing temporal patterns in the sales data
def create_lagged_features(data, lag=1):
    lagged_data = data.copy()
    for i in range(1, lag+1):
        lagged_data[f'lag_{i}'] = lagged_data['Sales'].shift(i)
    return lagged_data

lag = 5  
sales_with_lags = create_lagged_features(data[['Order Date', 'Sales']], lag)

sales_with_lags = sales_with_lags.dropna()
sales_with_lags = sales_with_lags.reset_index(drop=True)

# Enhance feature engineering - add calendar features and rolling statistics
def enhance_feature(data):
    df = data.copy()
    
    # Extract calendar features
    df['dayofweek'] = df['Order Date'].dt.dayofweek
    df['month'] = df['Order Date'].dt.month
    df['quarter'] = df['Order Date'].dt.quarter
    df['year'] = df['Order Date'].dt.year
    
    # Add day of month
    df['day'] = df['Order Date'].dt.day
    
    # Is weekend feature
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Add rolling statistics with multiple windows
    df['rolling_mean_7d'] = df['Sales'].rolling(window=7, min_periods=1).mean()
    df['rolling_std_7d'] = df['Sales'].rolling(window=7, min_periods=1).std()
    df['rolling_mean_30d'] = df['Sales'].rolling(window=30, min_periods=1).mean()
    
    # Fill any NaN values created by rolling windows
    df = df.bfill()
    
    return df

sales_with_features = enhance_feature(sales_with_lags)

# Prepare data for training and testing
X = sales_with_features.drop(columns=['Order Date', 'Sales'])
y = sales_with_features['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the XGBoost model for sales forecasting
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
model_xgb.fit(X_train, y_train)

# Make predictions and evaluate the model performance
predictions_xgb = model_xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, predictions_xgb))
mae_xgb = mean_absolute_error(y_test, predictions_xgb)
mape_xgb = np.mean(np.abs((y_test - predictions_xgb) / np.maximum(1e-10, y_test))) * 100
r2_xgb = r2_score(y_test, predictions_xgb)

# Evaluation metrics - RMSE, MAE, MAPE, and R²
print(f"RMSE: {rmse_xgb:.2f}")
print(f"MAE: {mae_xgb:.2f}")
print(f"MAPE: {mape_xgb:.2f}%")
print(f"R²: {r2_xgb:.4f}")

# Visualize the results by plotting the actual vs predicted sales
# Get the corresponding dates for the test set
test_dates = sales_with_lags['Order Date'].iloc[X_test.index]

plt.figure(figsize=(12, 6))
plt.plot(test_dates, y_test, label='Actual Sales', color='red')
plt.plot(test_dates, predictions_xgb, label='Predicted Sales', color='green')
plt.title('Sales Forecasting using XGBoost')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()