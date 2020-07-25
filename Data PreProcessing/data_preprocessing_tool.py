# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:39:20 2020

@author: chakanc
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def main():
    
    dataset = pd.read_csv('Data.csv')
    print(dataset)
    
    X = dataset.iloc[:,:-1]  # dataset.iloc[:,0:2]
    y = dataset.iloc[:,-1]
    #y.columns = ['Purchased']

    # Fix missing data
    #imputerobj = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    #imputerobj.fit(X)
    
    imputerobj = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputerobj.fit(X.iloc[:, 1:3])
    
    X.iloc[:, 1:3] = imputerobj.transform(X.iloc[:, 1:3])
    
    #oe = OrdinalEncoder(categories='auto', dtype=np.int)
    #y.iloc[:] = oe.fit_transform(np.array(y.iloc[:]))
    
    #encoding data
    le = LabelEncoder()
    y = le.fit_transform(y)
    
        
    cat_features = ['Country']
    ct = ColumnTransformer(transformers=[('enc1', OneHotEncoder(dtype=np.int32), cat_features)], remainder='passthrough')
    X = np.array(ct.fit_transform(X))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
       
    sc = StandardScaler()
    X_train[:,3:] = sc.fit_transform(X_train[:,3:])
    X_test[:,3:] = sc.transform(X_test[:,3:])
    
    print(X_train)
    print(X_test)
    

main()