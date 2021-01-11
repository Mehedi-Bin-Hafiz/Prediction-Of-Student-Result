from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
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
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import LogisticRegressionCV,BayesianRidge
from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings("ignore")
#database
MainDatabase = pd.read_excel("Database.xlsx")
# base on database we will set iloc
x = MainDatabase.iloc[:, 0:5].values  #independent variables
x=x.astype(int)
print(x)
y = MainDatabase.iloc[ : , -3].values #dependent variables
print(y)

thirtypercent=0.30  # training size 70%
fourtypercent=0.40   # training size 60%
fiftypercent=0.50    # training size 50%
sixtypercent=0.60    # training size 40%
seventypercent=0.70   # training size 30%

#naive bayes
print("\n########## Gradient Boosting ###########")
print("30% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=.30, random_state=0)
clf_gb = GradientBoostingRegressor(n_estimators=200)
clf_gb.fit(X_train,y_train)
y_pred = clf_gb.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("40% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=.40, random_state=0)
clf_gb = GradientBoostingRegressor(n_estimators=200)
clf_gb.fit(X_train,y_train)
y_pred = clf_gb.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("50% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=.50, random_state=0)
clf_gb = GradientBoostingRegressor(n_estimators=200)
clf_gb.fit(X_train,y_train)
y_pred = clf_gb.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("60% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=.60, random_state=0)
clf_gb = GradientBoostingRegressor(n_estimators=200)
clf_gb.fit(X_train,y_train)
y_pred = clf_gb.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("70% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=.70, random_state=0)
clf_gb = GradientBoostingRegressor(n_estimators=200)
clf_gb.fit(X_train,y_train)
y_pred = clf_gb.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



print('################(Neural Network)MLP regressor ################')
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=thirtypercent, random_state=0)

print("30% data usage rate")
MLPreg = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
y_pred = MLPreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("40% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=fourtypercent, random_state=0)
MLPreg = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
y_pred = MLPreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("50% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=fiftypercent, random_state=0)
MLPreg = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
y_pred = MLPreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("60% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=sixtypercent, random_state=0)
MLPreg = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
y_pred = MLPreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("70% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=seventypercent, random_state=0)
MLPreg = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
y_pred = MLPreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))




print('################(Neural Network)MLP regressor ################')
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=thirtypercent, random_state=0)

print("30% data usage rate")
DTReg = DecisionTreeRegressor()
DTReg.fit(X_train,y_train)
y_pred = DTReg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("40% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=fourtypercent, random_state=0)

DTReg = DecisionTreeRegressor()
DTReg.fit(X_train,y_train)
y_pred = DTReg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("50% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=fiftypercent, random_state=0)
DTReg = DecisionTreeRegressor()
DTReg.fit(X_train,y_train)
y_pred = DTReg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("60% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=sixtypercent, random_state=0)
DTReg = DecisionTreeRegressor()
DTReg.fit(X_train,y_train)
y_pred = DTReg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("70% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=seventypercent, random_state=0)
DTReg = DecisionTreeRegressor()
DTReg.fit(X_train,y_train)
y_pred = DTReg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


print('################ SVM regressor ################')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=thirtypercent, random_state=0)

print("30% data usage rate")
svmreg = svm.SVR()
svmreg.fit(X_train,y_train)
y_pred = svmreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("40% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=fourtypercent, random_state=0)

svmreg = svm.SVR()
svmreg.fit(X_train,y_train)
y_pred = svmreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("50% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=fiftypercent, random_state=0)
svmreg = svm.SVR()
svmreg.fit(X_train,y_train)
y_pred = svmreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("60% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=sixtypercent, random_state=0)
svmreg = svm.SVR()
svmreg.fit(X_train,y_train)
y_pred = svmreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("70% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=seventypercent, random_state=0)
svmreg = svm.SVR()
svmreg.fit(X_train,y_train)
y_pred = svmreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


print('################ Logistic Regression ################')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=thirtypercent, random_state=0)

print("30% data usage rate")
logreg = TweedieRegressor(power=1, alpha=0.5, link='log')
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("40% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=fourtypercent, random_state=0)

logreg = TweedieRegressor(power=1, alpha=0.5, link='log')
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("50% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=fiftypercent, random_state=0)
logreg = TweedieRegressor(power=1, alpha=0.5, link='log')
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("60% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=sixtypercent, random_state=0)
logreg = TweedieRegressor(power=1, alpha=0.5, link='log')
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print("70% data usage rate")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=seventypercent, random_state=0)
logreg = TweedieRegressor(power=1, alpha=0.5, link='log')
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
