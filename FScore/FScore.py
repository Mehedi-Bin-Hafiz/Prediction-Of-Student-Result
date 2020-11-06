import pandas as pd
from nltk.corpus import stopwords
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm.classes import OneClassSVM
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC

from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")


MainDatabase = pd.read_excel('../Accuracy/Database.xlsx')
#print(MainDatabase.head())
# base on database we will set iloc
x = MainDatabase.iloc[:, :5].values  #independent variables
print(x)
y = MainDatabase['Final'].values #dependent variables
y=y.astype('int') ### note y must be integer all time other wise ValueError: Unknown label type: 'continuous' is produced.
# print(y)
#datauserate



thirtypercent= 0.30  # training size 70%
fourtypercent= 0.40   # training size 60%
fiftypercent= 0.50    # training size 50%
sixtypercent= 0.60    # training size 40%
seventypercent= 0.70   # training size 30%




#knn

print("########## KNN algorithm ###########")

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=thirtypercent, random_state=0)
knn=KNeighborsClassifier(n_neighbors=3,p=2)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=30, FScore = {0:.2f}".format(100*score),"%")


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=fourtypercent, random_state=0)
knn=KNeighborsClassifier(n_neighbors=3,p=2)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=40, FScore = {0:.2f}".format(100*score),"%")


X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=fiftypercent, random_state=0)
knn=KNeighborsClassifier(n_neighbors=3,p=2)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=50, FScore = {0:.2f}".format(100*score),"%")


X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=sixtypercent, random_state=0)
knn=KNeighborsClassifier(n_neighbors=3,p=2)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=60, FScore = {0:.2f}".format(100*score),"%")


X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=seventypercent, random_state=0)
knn=KNeighborsClassifier(n_neighbors=3,p=2)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=70, FScore = {0:.2f}".format(100*score),"%")



#naive bayes
print("\n########## Naive Bayes algorithm ###########")
gnb = GaussianNB()

X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=thirtypercent, random_state=0)
gnb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=30, FScore = {0:.2f}".format(100*score),"%")


X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=fourtypercent, random_state=0)
gnb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=40, FScore = {0:.2f}".format(100*score),"%")

X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=fiftypercent, random_state=0)
gnb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=50, FScore = {0:.2f}".format(100*score),"%")

X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=sixtypercent, random_state=0)
gnb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=60, FScore = {0:.2f}".format(100*score),"%")

X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=seventypercent, random_state=0)
gnb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=70, FScore = {0:.2f}".format(100*score),"%")


print("\n########## Decision tree algorithm ###########")

dtc = DecisionTreeClassifier()
X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=thirtypercent, random_state=0)
clf = dtc.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=30, FScore = {0:.2f}".format(100*score),"%")


X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=fourtypercent, random_state=0)
clf = dtc.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=40, FScore = {0:.2f}".format(100*score),"%")


X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=fiftypercent, random_state=0)
clf = dtc.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=50, FScore = {0:.2f}".format(100*score),"%")

X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=sixtypercent, random_state=0)
clf = dtc.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=60, FScore = {0:.2f}".format(100*score),"%")

X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=seventypercent, random_state=0)
clf = dtc.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=70, FScore = {0:.2f}".format(100*score),"%")


print("\n########## SVM algorithm ###########")

clf = svm.SVC(kernel='linear') # Linear Kernel
X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=thirtypercent, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=30, FScore = {0:.2f}".format(100*score),"%")

X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=fourtypercent, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=40, FScore = {0:.2f}".format(100*score),"%")

X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=fiftypercent, random_state=0)
y_pred = clf.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=50, FScore = {0:.2f}".format(100*score),"%")


X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=sixtypercent, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=60, FScore = {0:.2f}".format(100*score),"%")

X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=seventypercent, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=70, FScore = {0:.2f}".format(100*score),"%")




print("\n########## Random Forest Algorithm ###########")
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=thirtypercent, random_state=0)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=30, FScore = {0:.2f}".format(100*score),"%")


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=fourtypercent, random_state=0)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=40, FScore = {0:.2f}".format(100*score),"%")


X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=fiftypercent, random_state=0)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=50, FScore = {0:.2f}".format(100*score),"%")


X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=sixtypercent, random_state=0)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=60, FScore = {0:.2f}".format(100*score),"%")


X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=seventypercent, random_state=0)
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=70, FScore = {0:.2f}".format(100*score),"%")


print("\n########## Neural Network algorithm ###########")

mpl = MLPClassifier(max_iter=1000,alpha=1,random_state=0)
X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=thirtypercent, random_state=0)
mpl.fit(X_train, y_train)
y_pred = mpl.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=30, FScore = {0:.2f}".format(100*score),"%")


X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=fourtypercent, random_state=0)
mpl.fit(X_train, y_train)
y_pred = mpl.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=40, FScore = {0:.2f}".format(100*score),"%")


X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=fiftypercent, random_state=0)
mpl.fit(X_train, y_train)
y_pred = mpl.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=50, FScore = {0:.2f}".format(100*score),"%")


X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=sixtypercent, random_state=0)
mpl.fit(X_train, y_train)
y_pred = mpl.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=60, FScore = {0:.2f}".format(100*score),"%")

X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=seventypercent, random_state=0)
mpl.fit(X_train, y_train)
y_pred = mpl.predict(X_test)
score=f1_score(y_test, y_pred, average='weighted')
print("test size=70, FScore = {0:.2f}".format(100*score),"%")




###vvi## for logistic
# Logistic = LogisticRegressionCV(cv=5,scoring='accuracy',random_state=0,n_jobs=-1,verbose=3,max_iter=100)

#################GradientBoostingRegressor#############

