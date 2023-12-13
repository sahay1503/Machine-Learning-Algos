import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data={
    'Age':[25,30,35,40,45,50,55,60],
    'Years_of_experience':[2,5,8,10,12,15,18,20],
    'Salary':[50000,60000,75000,80000,90000,100000,110000,120000]
}
df=pd.DataFrame(data)
x= df[['Age','Years_of_experience']]
y=df['Salary']

model=LinearRegression()

#fit the model on the data
model.fit(x,y)
new_employees_details=[[30,5]]
predicted_salary=model.predict(new_employees_details)
print("Predicted Salary for new employee details:")
print(f"Age:{new_employees_details[0][0]}, Years_of_experience:{new_employees_details[0][1]},Predicted Salary:{predicted_salary[0]:.2f}")
