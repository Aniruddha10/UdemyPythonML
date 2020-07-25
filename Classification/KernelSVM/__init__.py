#
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as mp


dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values
m = len(X)
n=len(dataset.columns)
print(X)
print(y)
print(m)
print(n)

#split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)

m_train = len(X_train)
m_test = len(X_test)

#feature scaling
X_train = np.array(X_train).reshape((m_train,2))
X_test = np.array(X_test).reshape((m_test,2))
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print(X_train)
print(X_test)

# y_train = np.array(y_train).reshape((m_train,1))
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)
# print(y_train)

svcclassifier = SVC(kernel='rbf', random_state=0)
svcclassifier.fit(X_train, y_train)
# prms = logisticRegressor.get_params()
# for prm in prms:
#     print(prm)

print(X_test[0,:])
print(sc_X.inverse_transform((X_test[0,:])))
y_testsingle_predict = svcclassifier.predict([X_test[0,:]])
print(y_testsingle_predict)

y_test_predict = svcclassifier.predict(X_test)
print(y_test_predict)
comparray = np.concatenate((y_test_predict.reshape(m_test,1), 
                            y_test.reshape(m_test,1)), axis=1)

# precision
diff = y_test_predict.reshape(m_test,1) - y_test.reshape(m_test,1)
print(diff)
errorcnt = 0
for d in diff:
    if(d != 0):
        errorcnt += 1            
print(errorcnt)
accuracy = ((m_test-errorcnt)/m_test) * 100
print(accuracy)


tn, fp, fn, tp = confusion_matrix(y_test.reshape(m_test,1), y_test_predict.reshape(m_test,1)).ravel()
print(tn)
print(fp)
print(fn)
print(tp)
precision = tp / (tp+fp)
recall = tp / (tp+fn)
print(precision)
print(recall)
print(accuracy_score(y_test, y_test_predict))








