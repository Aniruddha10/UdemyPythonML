#

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as mp

#read data
dataset = pd.read_csv('Position_Salaries.csv')
m = len(dataset)
n=len(dataset.columns)

X = dataset.iloc[:,1]
y = dataset.iloc[:,(n-1)]

scX = StandardScaler()
X = scX.fit_transform(np.array(X).reshape(m,1))

scy = StandardScaler()
y = scy.fit_transform(np.array(y).reshape(m,1))

print(X)
print(y)

sv_regressor = SVR(kernel='rbf')
sv_regressor.fit(X,y)


scale_tobe_predicted = scX.transform([[6.5]])
yscale_single_predict = sv_regressor.predict(scale_tobe_predicted)
y_single_predict = scy.inverse_transform(yscale_single_predict)
print(y_single_predict)

#y_predict = sv_regressor.predict(X)
#mp.scatter(scX.inverse_transform(X), scy.inverse_transform(y), color='blue')
#mp.plot(scX.inverse_transform(X),scy.inverse_transform(y_predict), color='red')
#mp.title('SVR Test')
#mp.xlabel('Level')
#mp.ylabel('Salary')



X_grid = np.arange(min(scX.inverse_transform(X)), max(scX.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
mp.scatter(scX.inverse_transform(X), scy.inverse_transform(y), color='blue')
mp.plot(X_grid,
        scy.inverse_transform(
            sv_regressor.predict(scX.transform(X_grid))), 
        color='red')
mp.title('SVR Test')
mp.xlabel('Level')
mp.ylabel('Salary')
mp.show()