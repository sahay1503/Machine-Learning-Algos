import pandas as pd

from sklearn.model_selection import train_test_split from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report

#Load the MNIST dataset

mnist_dataset_path='/content/drive/MyDrive/ML/cell_samples.csv' df pd.read_csv(mnist_dataset_path)

#Replace?" with NaN

df.replace('?", np.nan, inplace=True)

df.dropna(inplace = True)

#Get feature names

feature_names = df.columns.tolist()

# Extract features (X) and target variable (y)

X = df.drop(feature_names[-1], axis=1) # Features 
y = df[feature_names[-1]] # Target variable

# Split the dataset into training and testing sets.

X_train, X_test, y_train, y_test =train_test_split(x, y, test_size=0.2,random_state=42);

#Build the Decision Tree model

dt_classifier DecisionTreeClassifier(random_state=42)

#Train the model

dt_classifier.fit(X_train, y_train)

#Make predictions on the test set

y_pred dt_classifier.predict(X_test)

#Evaluate the model

accuracy=accuracy_score(y_test, y_pred) 
report=classification_report(y_test, y_pred)
print(f'Accuracy: (accuracy}')

print("\nClassification Report:\n', report)
