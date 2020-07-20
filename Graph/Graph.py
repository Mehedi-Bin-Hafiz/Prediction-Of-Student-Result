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
# base on database we will set iloc
x = MainDatabase.iloc[:, 0:5].values  #independent variables
# print(x)
y = MainDatabase['Final'].values #dependent variables
y=y.astype('int')
# print(y)
#datauserate


""" ######It is the brain of graph#####"""

############Selected Algo###########
clf = svm.SVC(kernel='linear') # Linear Kernel
X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=0.30, random_state=0)
clf.fit(X_train, y_train)


predictedFinal=[]
RealFinal=[]
RealSgpa=[]
RealMark = pd.read_excel(r'RealMark.xlsx')
# print(RealMark.head())

for ind in RealMark.index:
    pred = clf.predict([[RealMark['Quiz'][ind], RealMark['Attendance'][ind],RealMark['Presentation'][ind],RealMark['Assignment'][ind],RealMark['Mid'][ind]]])
    predictedFinal.append(pred.item())
    RealFinal.append((RealMark['Final'][ind]).round(decimals=2))
    RealSgpa.append((RealMark['SGPA'][ind]).round(decimals=2))

""" ###### Brain End #####"""


print(predictedFinal)
print(RealFinal)


########### vvi code for paper#########
plt.rcParams.update({'font.size': 8})
plt.rcParams["font.family"] = "Times New Roman"


#############  Line graph  ##############


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


########################  Bar graph  #######################
RealMark.insert(6,'PredictedMark',predictedFinal)
Sumdata=RealMark[['Quiz',"Attendance",'Presentation','Assignment','Mid','PredictedMark']].astype(float).sum(1)
RealMark.insert(7,'PredictedTotal',Sumdata)

PredictSgpa=[]
for ind in RealMark.index:
    element=RealMark['PredictedTotal'][ind]
    sgpa=(element*0.05).round(decimals=2)
    PredictSgpa.append(sgpa)

Real4=0
Real375=0
Real350=0
Real325=0
Real3=0

Real275=0
Real250=0
Real225=0
Real2=0

Real175=0
Real150=0
Real125=0
Real1=0

Real0=0

pre4=0
pre375=0
pre350=0
pre325=0
pre3=0

pre275=0
pre250=0
pre225=0
pre2=0

pre175=0
pre150=0
pre125=0
pre1=0

pre0=0


for  i in range(len(RealSgpa)):
    if RealSgpa[i]==4:
        Real4 += 1
    elif RealSgpa[i]>=3.75 and RealSgpa[i]<4:
        Real375 +=1
    elif RealSgpa[i]>=3.50 and RealSgpa[i]<3.75:
        Real350 +=1
    elif RealSgpa[i]>=3.25 and RealSgpa[i]<3.50:
        Real325 +=1
    elif RealSgpa[i]>=3.00 and RealSgpa[i]<3.25:
        Real3 +=1
    elif RealSgpa[i]>=2.75 and RealSgpa[i]<3.00:
        Real275 +=1
    elif RealSgpa[i]>=2.50 and RealSgpa[i]<2.75:
        Real250 +=1
    elif RealSgpa[i]>=2.25 and RealSgpa[i]<2.50:
        Real225 +=1
    elif RealSgpa[i]>=2.00 and RealSgpa[i]<2.25:
        Real2 +=1
    elif RealSgpa[i] >= 1.75 and RealSgpa[i] < 2.00:
        Real175 += 1
    elif RealSgpa[i] >= 1.50 and RealSgpa[i] < 1.75:
        Real150 += 1
    elif RealSgpa[i] >= 1.25 and RealSgpa[i] < 1.50:
        Real125 += 1
    elif RealSgpa[i] >= 0 and RealSgpa[i] < 1.25:
        Real0 += 1

for  i in range(len(PredictSgpa)):
    if PredictSgpa[i]==4:
        pre4 += 1
    elif PredictSgpa[i]>=3.75 and PredictSgpa[i]<4:
        pre375 +=1
    elif PredictSgpa[i]>=3.50 and PredictSgpa[i]<3.75:
        pre350 +=1
    elif PredictSgpa[i]>=3.25 and PredictSgpa[i]<3.50:
        pre325 +=1
    elif PredictSgpa[i]>=3.00 and PredictSgpa[i]<3.25:
       pre3 +=1
    elif PredictSgpa[i]>=2.75 and PredictSgpa[i]<3.00:
        pre275 +=1
    elif PredictSgpa[i]>=2.50 and PredictSgpa[i]<2.75:
        pre250 +=1
    elif PredictSgpa[i]>=2.25 and PredictSgpa[i]<2.50:
        pre225 +=1
    elif PredictSgpa[i]>=2.00 and PredictSgpa[i]<2.25:
        Real2 +=1
    elif PredictSgpa[i] >= 1.75 and PredictSgpa[i] < 2.00:
        pre175 += 1
    elif PredictSgpa[i] >= 1.50 and PredictSgpa[i] < 1.75:
        pre150 += 1
    elif PredictSgpa[i] >= 1.25 and PredictSgpa[i] < 1.50:
        pre125 += 1
    elif PredictSgpa[i] >= 0 and PredictSgpa[i] < 1.25:
        pre0 += 1


print(PredictSgpa)
print('pre',pre4)
print('real',Real4)

Real=[Real4,Real375,Real350,Real325,Real3,Real275,Real250,Real225,Real2,Real175,Real150,Real125,Real1]
predict=[pre4,pre375,pre350,pre325,pre3,pre275,pre250,pre225,pre2,pre175,pre150,pre125,pre1]
labels=['4','3.75','3.50','3.25','3.00','2.75','2.50','2.25','2.00','1.75','1.50','1.25','1']
x = np.arange(len(labels))
width=0.36
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, Real, width, label='Real')
rects2 = ax.bar(x + width/2, predict, width, label='Predict')
predictdata = [Real,predict]
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.grid()
plt.show()


########################pie Chart of full graph#####################

RealSgpa = MainDatabase['SGPA'].values

Real4=0
Real375=0
Real350=0
Real325=0
Real3=0

Real275=0
Real250=0
Real225=0
Real2=0

Real175=0
Real150=0
Real125=0
Real1=0

Real0=0

for  i in range(len(RealSgpa)):
    if RealSgpa[i]==4:
        Real4 += 1
    elif RealSgpa[i]>=3.75 and RealSgpa[i]<4:
        Real375 +=1
    elif RealSgpa[i]>=3.50 and RealSgpa[i]<3.75:
        Real350 +=1
    elif RealSgpa[i]>=3.25 and RealSgpa[i]<3.50:
        Real325 +=1
    elif RealSgpa[i]>=3.00 and RealSgpa[i]<3.25:
        Real3 +=1
    elif RealSgpa[i]>=2.75 and RealSgpa[i]<3.00:
        Real275 +=1
    elif RealSgpa[i]>=2.50 and RealSgpa[i]<2.75:
        Real250 +=1
    elif RealSgpa[i]>=2.25 and RealSgpa[i]<2.50:
        Real225 +=1
    elif RealSgpa[i]>=2.00 and RealSgpa[i]<2.25:
        Real2 +=1


sizes=Real4,Real375,Real350,Real325,Real3,Real275,Real250,Real225,Real2
explode = (0.1,0.017,0.017,0.017,0.017,0.017,0.017,0.017,0.017)
labels=['4','3.75','3.50','3.25','3.00','2.75','2.50','2.25','2.00']
#autopact show percentage inside graph
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',)
plt.axis('equal')
plt.show()
