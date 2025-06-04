# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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

# Enhance feature engineering - add calendar features and rolling statistics, or advanced features
def enhance_features(data, advanced=True):
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
    
    # For advanced features
    if advanced:
        # Growth rates
        df['growth_1d'] = df['Sales'].pct_change(1)
        df['growth_7d'] = df['Sales'].pct_change(7)
        
        # Exponential moving averages
        df['ema_7d'] = df['Sales'].ewm(span=7, min_periods=1).mean()
        df['ema_14d'] = df['Sales'].ewm(span=14, min_periods=1).mean()
        
        # Interaction features
        df['month_weekend'] = df['month'] * df['is_weekend']
    
    # Fill any NaN values
    df = df.bfill().ffill()
    
    return df

sales_with_features = enhance_features(sales_with_lags, advanced=True)

# Prepare data for training and testing
X = sales_with_features.drop(columns=['Order Date', 'Sales'])
y = sales_with_features['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the XGBoost model for sales forecasting with hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0]
}

print("Starting hyperparameter tuning...")
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(objective='reg:squarederror'),
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Make predictions and evaluate the model performance
predictions_xgb = best_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, predictions_xgb))
mae_xgb = mean_absolute_error(y_test, predictions_xgb)
mape_xgb = np.mean(np.abs((y_test - predictions_xgb) / np.maximum(1e-10, y_test))) * 100
r2_xgb = r2_score(y_test, predictions_xgb)

# Evaluation metrics - RMSE, MAE, MAPE, and R²
print(f"RMSE: {rmse_xgb:.2f}")
print(f"MAE: {mae_xgb:.2f}")
print(f"MAPE: {mape_xgb:.2f}%")
print(f"R²: {r2_xgb:.4f}")

# Visualize feature importance for analysis
plt.figure(figsize=(10, 6))
xgb.plot_importance(best_model, max_num_features=10)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

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