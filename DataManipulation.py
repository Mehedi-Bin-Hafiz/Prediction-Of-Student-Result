import pandas as pd
import numpy as np
maindata=pd.read_excel(r'Main_data.xls',index=False).fillna(0)

M1=maindata.iloc[:, 3:-2].values

mid1=[]
for i in range(0,len(M1)):
    mid1.append(M1.item(i))
M2=maindata.iloc[:, 4:-1].values
mid2=[]
for i in range(0,len(M2)):
    mid2.append(M2.item(i))

Mid=[]
for i in range(len(mid1)):
     if(mid1[i]>mid2[i]):
         Mid.append(mid1[i])
     else:
         Mid.append(mid2[i])



GPA=maindata.iloc[:, 5: ].values
TotalMark=[]
for i in range(0,len(GPA)):
    TotalMark.append(GPA.item(i)/0.05)

Quiz=((maindata[['Q1','Q2','Q3']].astype(float).sum(1))/3).apply(np.ceil)
maindata.insert(3,"Quiz",Quiz,True)
maindata.insert(6,"Mid",Mid,True)
maindata.insert(7,"TotalMark",TotalMark,True)
finaldata=maindata[['Quiz','Mid','CGPA','TotalMark']]

# finaldata.to_excel(r'QuizMidSgpa.xlsx', index = False)
latestdata=finaldata.loc[finaldata['Quiz'] != 0 ]
latestdata2=latestdata.loc[latestdata['Mid'] != 0]


print(latestdata2)
# latestdata.to_excel(r'Path where you want to store the exported excel file\File Name.xlsx', index = False)

"""QIZ MID GPA and TotalMark found"""



T=latestdata2.iloc[:, 3: ].values
tm=T.item(0)
p=tm/100
at=7*p
print(at)
pr=8*p
print(pr)
ass=5*p
print(ass)
fin=tm-(at+pr+ass+9+18.80)
print(fin)
tota=at+pr+ass+fin+9+18.80
print(tota)
