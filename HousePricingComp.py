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
#print(train_data.columns)

#target
y = train_data.SalePrice

#features
features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
            'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 
            'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 
            'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
            'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

X = train_data[features]

#splitting data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

#model
HousePriceRF_model = RandomForestRegressor(random_state = 1)

#fitting model
HousePriceRF_model.fit(train_X, train_y)

#predicting
HousePriceRF_predictions = HousePriceRF_model.predict(val_X)

#mean absolute error
HousePriceRF_mae = mean_absolute_error(val_y, HousePriceRF_predictions)
print("Mean Absolute Error (Random Forest): ", HousePriceRF_mae)

#final model
test_data_path = r"C:\Users\sms20\Machine Learning\Kaggle-Housing-Prices-Competition\Housing-Prices-Competition\test.csv"
test_data = pd.read_csv(test_data_path)

test_X = test_data[features]

final_model = RandomForestRegressor(random_state = 1)

final_model.fit(X, y)

final_predictions = HousePriceRF_model.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': final_predictions})
output.to_csv('submission.csv', index = False)




# #Using DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

test_model = DecisionTreeRegressor(random_state = 1)

test_model.fit(train_X, train_y)

test_predictions = test_model.predict(val_X)

test_mae = mean_absolute_error(val_y, test_predictions)
print("Mean Absolute Error(Decision Tree): ", test_mae)



