from  sklearn import datasets
import pandas as pd
import numpy as np
iris=datasets.load_iris()
iris.target_names
iris.keys()
iris.data.shape
iris.data
iris=pd.DataFrame(data=np.c_[iris['data'],iris['target']],columns=iris['feature_names']+['target'])
iris.head()

species=[]
for i in range(len(iris['target'])):
  if iris['target'][i]==0:
    species.append("setosa")
  elif iris['target'][i]==1:
      species.append('versicolor')
  else:
    species.append('virginica')

iris['species']=species
import matplotlib.pyplot as plt
setosa=iris[iris.species=="setosa"]
versicolor=iris[iris.species=="versicolor"]
virginica=iris[iris.species=="virginica"]

fig,ax=plt.subplots()
fig.set_size_inches(10,7)
ax.scatter(setosa['petal length (cm)'],setosa['petal width (cm)'],label='Setosa Petal',facecolor="blue")
ax.scatter(versicolor['petal length (cm)'],versicolor['petal width (cm)'],label='Versicolor',facecolor="green")
ax.scatter(virginica['petal length (cm)'],virginica['petal width (cm)'],label='Virginica',facecolor="red")
ax.set_xlabel('Sepal length(cm)')
ax.set_ylabel('Sepal width (cm)')
ax.grid()
ax.legend()

from sklearn.model_selection import train_test_split
X = iris.drop(['sepal length (cm)','sepal width (cm)','target','species'],axis=1)
y =iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=82)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

kernels=['linear','rbf','poly']
for kernel in kernels:
  model = SVC(kernel=kernel)
  model.fit(X_train,y_train)
  pred=model.predict(X_test)
  print("acuracy using {}:".format(kernel),accuracy_score(pred,y_test))
