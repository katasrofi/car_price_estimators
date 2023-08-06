import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_data = pd.read_csv('train.csv')
dataset = raw_data.copy()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
dataset = imputer.fit_transform(dataset)

brand = pd.get_dummies(dataset[:, 0])
fuel_type = pd.get_dummies(dataset[:, 8])
drivetrain = pd.get_dummies(dataset[:, 9])

categorical = [0, 1, 4, 6, 8, 9, 33, 34]
dataset = np.delete(dataset, categorical, axis=1)
dataset[:, -1] = np.where(dataset[:, -1] == 'ot Priced', '0', dataset[:, -1])
price = dataset[:, -1:]

dataset = np.concatenate((dataset, brand, fuel_type, drivetrain, price), axis = 1)
    
# Split data to x and y
x = dataset[:, :-1]
y = dataset[:, -1].astype(int)

from sklearn.model_selection import train_test_split, cross_val_score
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import Lasso
lasso = Lasso(random_state=1)
lasso.fit(x_train, y_train)

y_pred = lasso.predict(x_test)
result = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_pred),1)), axis =1)

# Accuracy
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("MSE pada data uji:", mse)

# Cross validation
scores = cross_val_score(lasso, x_train, y_train, cv=3, scoring='neg_mean_absolute_error')

mean_mse = -np.mean(scores)
print('Mean MSE: ', mean_mse)

# Learning Curve
from sklearn.model_selection import learning_curve

# Buat fungsi untuk plot kurva pembelajaran
def plot_learning_curve(model, x, y):
    train_sizes, train_scores, val_scores = learning_curve(model, x, y, cv=5,
                                                           scoring='neg_mean_squared_error')
    train_scores_mean = -np.mean(train_scores, axis=1)
    val_scores_mean = -np.mean(val_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, val_scores_mean, label='Validation error')
    plt.xlabel('Training set size')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()

# Plot kurva pembelajaran
plot_learning_curve(lasso, x_train, y_train)

# Hitung residual pada data pelatihan
y_train_pred = lasso.predict(x_train)
residuals = y_train - y_train_pred

# Plot residual
plt.scatter(y_train_pred, residuals)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Test 
data = pd.read_csv('test.csv')
dataset_test = data.copy()

# Standardization
dataset_test = imputer.transform(dataset_test)

# Make dummies
brand_test = pd.get_dummies(dataset_test[:, 0])
fuel_type_test = pd.get_dummies(dataset_test[:, 8])
drivetrain_test = pd.get_dummies(dataset_test[:, 9])

# Drop data
categorical = [0, 1, 4, 6, 8, 9, 33, 34]
dataset_test = np.delete(dataset_test, categorical, axis=1)
dataset_test[:, -1] = np.where(dataset_test[:, -1] == 'ot Priced', '0', dataset_test[:, -1])
Price = dataset_test[:, -1:]

dataset_test = np.concatenate((dataset_test, brand_test, fuel_type_test, drivetrain_test, Price), axis = 1)
    
# Split data to x and y
X_test = dataset[:, :-1]
X_test = sc.transform(X_test)
Y_test = dataset[:, -1].astype(int)


Y_pred = lasso.predict(X_test)
result_test = np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_pred),1)), axis =1)

# Menghitung jumlah fitur dan jumlah sampel
num_features = x.shape[1]
num_samples = len(x)

# Menghitung nilai R-squared
r_squared = lasso.score(x_test, y_test)

# Menghitung adjusted R-squared
adjusted_r_squared = 1 - (1 - r_squared) * (num_samples - 1) / (num_samples - num_features - 1)
