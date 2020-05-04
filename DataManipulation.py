import pandas as pd
import numpy as np
import tabula



""""   Very Dangerous program. Never run it without permission  """""






# maindata=pd.read_excel(r'Main_data.xls',index=False).fillna(0)
#
# M1=maindata.iloc[:, 3:-2].values
#
# mid1=[]
# for i in range(0,len(M1)):
#     mid1.append(M1.item(i))
# M2=maindata.iloc[:, 4:-1].values
# mid2=[]
# for i in range(0,len(M2)):
#     mid2.append(M2.item(i))
#
# Mid=[]
# for i in range(len(mid1)):
#      if(mid1[i]>mid2[i]):
#          Mid.append(mid1[i])
#      else:
#          Mid.append(mid2[i])
#
# #############modifying quiz and mid mark###############
# Quiz=((maindata[['Q1','Q2','Q3']].astype(float).sum(1))/3).apply(np.ceil)
# maindata.insert(3,"Quiz",Quiz,True)
# maindata.insert(6,"Mid",Mid,True)
# ############# modifying quiz and mid mark ###############
#
# # print(maindata)
#
# ################# Generating attendence presentation assingment ###########
# GPA=maindata.iloc[:, 7: ].values
# TotalMark=[]
# for i in range(0,len(GPA)):
#     TotalMark.append(GPA.item(i)/0.05)
# # print(TotalMark)
# attendance=[]
# presentation=[]
# assignment=[]
# for i in TotalMark:
#     tm = i
#     p = tm / 100
#     attendance.append(7 * p)
#     presentation.append(8 * p)
#     assignment.append(5*p)
# maindata.insert(1,"Attendance",attendance,True)
# maindata.insert(2,'Presentation',presentation,True)
# maindata.insert(4,'Assignment',assignment,True)
# maindata.insert(5,'TotalMark',TotalMark,True)
# #################Generating attendence presentation assingment ###########
#
#
#
# finaldata=maindata[['Quiz',"Attendance",'Presentation','Assignment','Mid','TotalMark','CGPA']].round(decimals=2)
#
# # finaldata.to_excel(r'QuizMidSgpa.xlsx', index = False)
# latestdata=finaldata.loc[finaldata['Quiz'] != 0 ]
# latestdata2=latestdata.loc[latestdata['Mid'] != 0]
#
# print(latestdata2)
#
# #####latestdata2.to_excel(r'WithoutFinalData.xlsx', index = False) #should not run with out permission
#

# """ QIZ MID assignment presentation attendance GPA and TotalMark found """
#
#
# WFdatabase=pd.read_excel(r'WithoutFinalData.xlsx',index=False)
#
# Sumdata=WFdatabase[['Quiz',"Attendance",'Presentation','Assignment','Mid']].astype(float).sum(1)
#
# FinalMark= WFdatabase["TotalMark"]-Sumdata
# WFdatabase.insert(5,"Final",FinalMark,True)
# WFdatabase.to_excel(r'ManipulatedData.xlsx', index = False)


#
# for i in range(1,78):
#     df = "./PdfResult/{}.pdf".format(i)
#     output = "./CsvResult/{}.csv".format(i)
#     tabula.convert_into(df, output, output_format="csv", stream=True)

