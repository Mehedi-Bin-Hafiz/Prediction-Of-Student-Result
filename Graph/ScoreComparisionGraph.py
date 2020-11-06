import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score,recall_score
from sklearn.linear_model import LogisticRegressionCV
import warnings
warnings.filterwarnings("ignore")

Fsocrelis=list()
Precisionlis=list()
recalllis=list()
Accuracylis=list()


df = pd.read_excel('../Accuracy/Database.xlsx')





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

print("########## SVM algorithm ###########")

clf = svm.SVC(kernel='linear') # Linear Kernel
X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=thirtypercent, random_state=0)
clf.fit(X_train, y_train)




y_pred = clf.predict(X_test)
FScore=f1_score(y_test, y_pred, average='weighted')
AScore=clf.score(X_test,y_test)
PreScore=precision_score(y_test, y_pred, average='weighted')
ReScore=recall_score(y_test, y_pred, average='weighted')
print("test size=30, FScore = {0:.2f}".format(100*FScore),"%")
print("test size=30, AScore = {0:.2f}".format(100*AScore),"%")
print("test size=30, PreScore = {0:.2f}".format(100*PreScore),"%")
print("test size=30, ReScore = {0:.2f}".format(100*ReScore),"%")



###################Generation of Graph##################

objects = ( 'Accuracy', 'Precision', 'Recall','F1-Score')
y_pos = np.arange(len(objects))
performance = [AScore*100,PreScore*100,ReScore*100,FScore*100]
axes = plt.axes()
axes.set_yticks([0,  10,  20,  30, 40, 50, 60, 70, 80, 90, 100])
plt.grid()
plt.bar(y_pos, performance, align='center',width=0.40, color=['#ff9f43','#16a085','#ff6348','#575fcf'])
plt.xlabel('Score Matrix')
plt.ylabel('Score Matrix')
plt.xticks(y_pos, objects)
plt.show()