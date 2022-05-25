# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# k=pd.read_csv('C:\\Users\\Hp\\PycharmProjects\\Sample\\static\\heart.csv',sep=',',header=None, skiprows=1)
#
# x=k.values[:,0:13]
# print (x)
#
# y=k.values[:,13:14]
# print(y)
# x1=[[55,1,0,132,353,0,1,132,1,1.2,1,1,3]]
# obj=RandomForestClassifier()
# obj.fit(x,y)
# op=obj.predict(x1)
# print(op)

import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# k=pd.read_csv('D:\\riss\\python\\2020-2021\\project\\crop productivity\\web\\crop_yeild_prediction\\static\\cpdata.csv',sep=',',header=None, skiprows=1)

# X = [[0], [1], [2], [3]]
# y = [0, 0, 1, 1]
# X=k.values[:,0:13]
# print (X)
# print("------")
# y=k.values[:,13:14]
# # print("-----"+y)
# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(X, y)
# # KNeighborsClassifier(...)
# print("pppppp---"+neigh.predict([[1.1]]))
# # [0]
# # print(neigh.predict_proba([[0.9]]))
# # [[0.66666667 0.33333333]]

tp = float(25)
hd = float(36)
ph = float(2)
rf = float(5)
a=[[tp,hd,ph,rf]]
data=pd.read_csv("D:\\riss\\python\\2020-2021\\project\\crop productivity\\web\\crop_yeild_prediction\\static\\cpdata.csv")
atribute=data.iloc[1:,0:4].values
label=data.iloc[1:,4].values
# from sklearn.tree import DecisionTreeClassifier
# c=DecisionTreeClassifier()
# c.fit(atribute,label)
# p=c.predict(a)
neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(atribute, label)
# KNeighborsClassifier(...)
a=neigh.predict(a)
print("pppppp---"+a[0])
# print(p)