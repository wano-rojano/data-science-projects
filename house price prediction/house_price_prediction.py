# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np

dataset = pd.read_excel("HousePricePrediction.xlsx")

# Explore the dataset
print("Data shape:", dataset.shape)
print("\nData info:")
print(dataset.info())
print("\nData sample:")
print(dataset.head())

# Data preprocessing
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))

int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))

# Exploratory Data Analysis (EDA)
numerical_dataset = dataset.select_dtypes(include=['number'])

plt.figure(figsize=(12, 6))
sns.heatmap(numerical_dataset.corr(),
            cmap = 'BrBG',
            fmt = '.2f',
            linewidths = 2,
            annot = True)
plt.show()

unique_values = []
for col in object_cols:
    unique_values.append(dataset[col].unique().size)
plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols,y=unique_values)
plt.show()

plt.figure(figsize=(18, 36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1

for col in object_cols:
    y = dataset[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1
plt.show()

# Data cleaning
dataset.drop(['Id'],
            axis=1,
            inplace=True)

dataset['SalePrice'] = dataset['SalePrice'].fillna(
    dataset['SalePrice'].mean())

new_dataset = dataset.dropna()

new_dataset.isnull().sum()

# OneHotEncoder - for label categorical features
s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ', 
    len(object_cols))

OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

# Split the dataset into training and testing
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=0)

# Model training and accuracy evaluation using Support Vector Machine (SVM) with hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly'],
    'epsilon': [0.1, 0.2, 0.5]
}

print("Starting hyperparameter tuning...")
grid_search = GridSearchCV(
    svm.SVR(), 
    param_grid, 
    cv=5,
    scoring='neg_mean_absolute_error',
    verbose=1
)
grid_search.fit(X_train, Y_train)

best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Make predictions using the best model
Y_pred = best_model.predict(X_valid)

# Evaluate the model performance
mae = mean_absolute_error(Y_valid, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_valid, Y_pred))
mape = mean_absolute_percentage_error(Y_valid, Y_pred)
r2 = r2_score(Y_valid, Y_pred)

# Evaluation metrics - MAE, RMSE, MAPE, and R²
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Percentage Error: {mape:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualize the results by plotting the actual vs predicted house prices
plt.figure(figsize=(10, 6))
plt.scatter(Y_valid, Y_pred, alpha=0.7, label='Predictions')
plt.plot([Y_valid.min(), Y_valid.max()], [Y_valid.min(), Y_valid.max()], 'r--', 
        label='Perfect Predictions (Actual = Predicted)')

z = np.polyfit(Y_valid, Y_pred, 1)
plt.plot(Y_valid, np.poly1d(z)(Y_valid), "b-", 
        label=f'Trend Line (y={z[0]:.2f}x+{z[1]:.2f})')

plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices (Unscaled)')
plt.legend()
plt.tight_layout()
plt.show()

# Residual analysis
plt.scatter(Y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()