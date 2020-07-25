#

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as mp

def main():
    print('main function')
    dataset = pd.read_csv('Position_Salaries.csv')
    X = dataset.iloc[:,1:-1].values
    y = dataset.iloc[:,-1].values
    #print(X)
    #print(np.array(y).reshape((len(y),1)))
    #X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    
    regressor = LinearRegression()
    regressor.fit(X, y)
    y_lin_predict = regressor.predict(X)
    #print(y_predict)
    #print(np.concatenate((y_test.reshape(2,1), y_predict.reshape(2,1)),1))
    #print(X)
    polyFeature = PolynomialFeatures(degree=2)
    X_poly = polyFeature.fit_transform(X)
    #print(X_poly)
    
    linreg = LinearRegression()
    linreg.fit(X_poly, y)
    y_predict = linreg.predict(X_poly)
    
    mp.scatter(X,y, color='red')
    mp.plot(X, y_lin_predict, color='black')
    mp.plot(X, y_predict, color='blue')
    mp.title('Truth?')
    mp.xlabel('Position Label')
    mp.ylabel('Salary')
    mp.show()
    
    X_predict_lin_array = [[6.5]]
    y_predict_lin_array = regressor.predict(X_predict_lin_array)
    print(y_predict_lin_array)
    
    X_predict_array = [[1,6.5, 6.5*6.5]]
    y_predict_array = linreg.predict(polyFeature.fit_transform(X_predict_lin_array))
    print(y_predict_array)
    
    
    
main()