print("....Algorithm.....")
### Importing all libraries required
### Reading data from remote link
### Plotting the distribution of scores
### divide the data into "attributes" (inputs) and "labels" (outputs)
### split this data into training and test sets using using \n ---
### -- Scikit-Learn's built-in train_test_split() method
### Plotting the regression line and for test data
### calculate the mean square error

print(".....Linear Regression with Python Scikit learn....")

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from matplotlib import style

url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")
print(data.head(10))

data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values 

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
print("Training complete.")


line = regressor.coef_*X+regressor.intercept_

plt.scatter(X, y)
plt.plot(X, line)
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.title('Hours vs Percentage')
plt.show()

print(X_test)
y_predict = regressor.predict(X_test)

df = pd.DataFrame({'Actual' : y_test , 'Predicted' : y_predict})
df

hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))
from sklearn import metrics
print("Mean Absolute Error1:",metrics.mean_absolute_error(y_test,y_predict))
print("Mean Absolute Error2:" , metrics.mean_absolute_error(X_test,y_predict))
