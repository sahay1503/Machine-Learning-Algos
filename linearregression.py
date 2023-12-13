import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data= pd.read_csv("datasets/Salary_data.csv")
data.head(10)


x=df[['YearsExperience']]
y=df['Salary']
model=LinearRegression()
model.fit(x,y)
new_sizes=[[13], [15]]
predicted_prices=model.predict(new_sizes)
print('predicted prices :')
for year, salary in zip(new_sizes, predicted_prices):
  print(f"Plot Size: {year[0]}, Predicted price: {salary:.2f}")


plt.scatter(x,y, color='blue', label='Actual Prices')
plt.plot(x,model.predict(x), color='red', linewidth=2,label='Linear Regression')
plt.xlabel('YearsExperince')
plt.ylabel('Salary')
plt.legend()
plt.title("Linear regression")
plt.show()
