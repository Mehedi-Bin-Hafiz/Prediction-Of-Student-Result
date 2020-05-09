import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


MainDatabase = pd.read_excel(r'RealMark.xlsx')
print(MainDatabase.head())
# base on database we will set iloc
x = MainDatabase.iloc[:, :6].values  #independent variables
# print(x)
y = MainDatabase['TotalMark'].values #dependent variables
y=y.astype('int')
print(y)
#datauserate

############Selected Algo###########
dtc = DecisionTreeClassifier()
X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=0.40, random_state=0)
clf = dtc.fit(X_train,y_train)
#Predict the response for test dataset

pred = clf.predict(X_test)
