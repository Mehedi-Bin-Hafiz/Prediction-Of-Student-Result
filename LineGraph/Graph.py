import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier


MainDatabase = pd.read_excel(r'../Accuracy/Database.xlsx')
print(MainDatabase.head())
# base on database we will set iloc
x = MainDatabase.iloc[:, 0:5].values  #independent variables
# print(x)
y = MainDatabase['Final'].values #dependent variables
y=y.astype('int')
# print(y)
#datauserate

############Selected Algo###########
clf = svm.SVC(kernel='linear') # Linear Kernel
X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=0.30, random_state=0)
clf.fit(X_train, y_train)


predictedFinal=[]
RealFinal=[]
RealMark = pd.read_excel(r'RealMark.xlsx')
# print(RealMark.head())

for ind in RealMark.index:
    pred = clf.predict([[RealMark['Quiz'][ind], RealMark['Attendance'][ind],RealMark['Presentation'][ind],RealMark['Assignment'][ind],RealMark['Mid'][ind]]])
    predictedFinal.append(pred.item())
    RealFinal.append((RealMark['Final'][ind]).round())





################Line graph##############
XandYLen=[]
for i in range(1,len(RealFinal)+1):
    XandYLen.append(i)
axes = plt.axes()
plt.plot(XandYLen,RealFinal,color='red',linewidth=2)
plt.plot(XandYLen,predictedFinal,color='green',linewidth=2)
axes.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45,50])
plt.grid()
plt.legend(['Real Marks','Predicted Marks'])
plt.show()