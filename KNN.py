import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',names=['sepal_length1','sepal_width','petal_length','petal_width','species'])
# data.head()
x=data[['sepal_length1','sepal_width','petal_length','petal_width']]
y=data['species']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print("K-Nearest Neighbors Classifier accuracy",accuracy)
