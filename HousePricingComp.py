# code checking set up
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex7 import *

#file paths
import os
if not os.path.exists(r"C:\Users\sms20\Machine Learning\Kaggle-Housing-Prices-Competition\Housing-Prices-Competition\train.csv"):
    os.symlink(r"C:\Users\sms20\Machine Learning\Kaggle-Housing-Prices-Competition\Housing-Prices-Competition\train.csv", r"C:\Users\sms20\Machine Learning\Kaggle-Housing-Prices-Competition\Housing-Prices-Competition\train.csv")  
    os.symlink(r"C:\Users\sms20\Machine Learning\Kaggle-Housing-Prices-Competition\Housing-Prices-Competition\test.csv", r"C:\Users\sms20\Machine Learning\Kaggle-Housing-Prices-Competition\Housing-Prices-Competition\test.csv")  
    print("Symlink created")

# libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#loading data
train_data = pd.read_csv(r"C:\Users\sms20\Machine Learning\Kaggle-Housing-Prices-Competition\Housing-Prices-Competition\train.csv")
print(train_data.columns)

#target
y = train_data.SalePrice

#features
features = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',     
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',    
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',    
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',       
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',    
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',        
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']

X = train_data[features]

#one-hot encoding
one_hot_X = pd.get_dummies(X)

#splitting data
train_X, val_X, train_y, val_y = train_test_split(one_hot_X, y, random_state = 1)

#model
HousePriceRF_model = RandomForestRegressor(random_state = 1)

#fitting model
HousePriceRF_model.fit(train_X, train_y)

#predicting
HousePriceRF_predictions = HousePriceRF_model.predict(val_X)

#mean absolute error
HousePriceRF_mae = mean_absolute_error(val_y, HousePriceRF_predictions)
print("Mean Absolute Error: ", HousePriceRF_mae)




