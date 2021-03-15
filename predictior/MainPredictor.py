
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.linear_model import LogisticRegressionCV,BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

warnings.filterwarnings("ignore")
#database
MainDatabase = pd.read_excel("../Database/Database.xlsx").iloc[:14000]
# base on database we will set iloc
x = MainDatabase.iloc[: , :5].values  #independent variables
x = x.round(1)
print(x)
y = MainDatabase.iloc[ : , -1].values #dependent variables
print(y)

ValidationDataset = pd.read_excel("../Database/ValidationDataset.xlsx").loc[:1400]
Vx = ValidationDataset.iloc[: , :5].values
Vy = ValidationDataset.iloc[ : , -1].values


print('################ Linear Regression ################')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=0.3, random_state=0)
print("30% data usage rate")
clf_gb = GradientBoostingRegressor(n_estimators=200)
clf_gb.fit(X_train,y_train)
y_pred = clf_gb.predict(Vx)
print('Mean Absolute Error:', metrics.mean_absolute_error(Vy, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Vy, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Vy, y_pred)))
print('r2_sore:',r2_score(Vy,y_pred))

print(len(y_pred))
print(ValidationDataset['SGPA'].values.tolist())

axes = plt.axes()
XandYlen = [x for x in range(0,len(y_pred))]
plt.plot(XandYlen, ValidationDataset['SGPA'].values.tolist(), linewidth=7,color='#EBCA3B')
plt.plot(XandYlen, y_pred,  linewidth=2,color='green')
plt.yticks([1.50,1.75,2,2.25,2.50,2.75,3,3.25,3.50,3.75,4,4.25,4.50,4.75,5])
plt.grid()
plt.legend(['Real value', 'Predicted value'])
plt.xlabel('Numbers')
plt.ylabel('Prediction')
plt.savefig("real vs prediction.png")
plt.show()
