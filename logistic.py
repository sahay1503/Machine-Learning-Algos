import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

# Read the CSV file
data = pd.read_csv("/content/Marketingcampaigns.csv")

# Explore the dataset (optional)
# print(df.head())

data.shape

data=pd.get_dummies(data,columns=['Location','Gender'])

data.head()
d=data.isnull()

df=pd.DataFrame(d)
df.to_csv("missing.csv",index=False)
data=data.dropna()
data.shape

x=data.drop('Location',axis=1)
y=data['Location']

x.head()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Create and train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
f1score=f1_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
print(accuracy)
print(f1score)
print(precision)
print(recall)

ax = sns.heatmap(confusion, annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()
