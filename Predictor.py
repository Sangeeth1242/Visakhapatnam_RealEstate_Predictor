import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import glob


def predict_price(location, year, total_sqft, bath, bhk):
    # import data and modify
    path = r'C:\\Users\\sange\\Documents\\GitHub\\Visakhapatnam_RealEstate_Predictor'
    filenames = glob.glob(path + "\*.xlsx")
    data = pd.DataFrame()
    for file in filenames:
        df = pd.concat(pd.read_excel(file, sheet_name=None),
                       ignore_index=True, sort=False)
        data = data._append(df, ignore_index=True)
    data.to_csv(r'finaldata.csv', index=False)
    data_set = pd.read_csv('finaldata.csv')
    x = data_set.iloc[:, :-2].values
    y = data_set.iloc[:, -1].values

    labelencoder_x = LabelEncoder()
    x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
    # onehotencoder = OneHotEncoder(categories=[0])
    # x = onehotencoder.fit_transform(x).toarray()

    # split data into train data and test data
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, random_state=0, test_size=.12)

    # initialise the data
    model = LinearRegression()
    model.fit(train_x, train_y)

    # make prediction
    y_prediction = model.predict([[location, year, total_sqft, bath, bhk]])
    print('price : ', y_prediction)

    # Accuracy of the model
    y_test_predivtion = model.predict(test_x)
    print('mean absolute error : ', mean_absolute_error(test_y, y_test_predivtion))
    print('r2 score : ', r2_score(test_y, y_test_predivtion))


data_set = pd.read_csv('data_ser.csv')
for i in pd.unique(data_set['location']):
    print(i)
location = int(input('enter location : '))
year = int(input('enter year : '))
sq_feet = int(input('enter square feet : '))
bath = int(input('enter number of bathrooms : '))
bed = int(input('enter numner of bed rooms : '))
predict_price(location, year, sq_feet, bath, bed)
