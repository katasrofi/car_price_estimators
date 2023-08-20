import pandas as pd
import numpy as np

dataset = pd.read_csv('train.csv')

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')

# Fill empty value with Imputer
columns_with_nan = dataset.columns[dataset.isnull().any()]
dataset[columns_with_nan] = imputer.fit_transform(dataset[columns_with_nan])

# Save dataset
dataset.to_csv('dataset_train', index = False)