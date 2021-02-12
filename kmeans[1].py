import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('Country-data.csv')
data

country=data.drop(['country'],axis=1,inplace=True)
data

correlation=data.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation,annot=True)

d1=data

sc=StandardScaler()
data=sc.fit_transform(data)

from sklearn.cluster import KMeans

cost_func=[]
for k in range(1,15):
  model=KMeans(n_clusters=k)
  model.fit(data)
  cost_func.append(model.inertia_)

plt.plot(list(range(1,15)),cost_func)

model=KMeans(n_clusters=3)
cluster=model.fit_predict(data)

#d1=pd.DataFrame(data)
d1['clusters']=cluster

d1

d1['clusters'].value_counts()

d1

sns.color_palette('tab10',3)

sns.pairplot(d1,hue='clusters',palette=sns.color_palette('tab10',3))

