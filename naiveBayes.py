import  numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report

iris=load_iris()
X=iris.data
y=iris.target

#split the databases
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

nb=MultinomialNB()
nb.fit(x_train,y_train)

y_pred=nb.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy}")

report=classification_report(y_test,y_pred,target_names=iris.target_names)
print("Classification report:\n",report)
