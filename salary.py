import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

datasets=pd.read_csv('Salary_Data.csv')
X=datasets.iloc[:,:1].values 
Y=datasets.iloc[:,1].values

#Now we will split the dataset into Test and Train 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=1/3, random_state=0)

#fitting Simple Linear Regression to the Training Set 
from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

#predicting the Test Set result 
Y_pred=regressor.predict(X_test)

#let's visualise the Training Set results 
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train),color='red')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

#let's visualise the Training Set results 
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train),color='red')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
