# EXPERIMENT-06

# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## Aim:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. PREPARE YOUR DATA:
   
  Clean and format your data

 Split your data into training and testing sets

2.DESIGN YOUR MODEL

  Use a sigmoid function to map inputs to outputs

  Initialize weights and bias terms
  
3.DEFINE YOUR COST FUNCTION  

  Use binary cross-entropy loss function

   Penalize the model for incorrect prediction
   
4.DEFINE YOUR LEARNING RATE 

  Determines how quickly weights are updated during gradient descent
  
5.TRAIN YOUR MODEL

  Adjust weights and bias terms using gradient descent

  Iterate until convergence or for a fixed number of iterations
  
6.EVALUATE YOUR MODEL 

   Test performance on testing data

   Use metrics such as accuracy, precision, recall, and F1 score
   
7.TUNE HYPERPARAMETERS

   Experiment with different learning rates and regularization techniques
   
8.DEPLOY YOUR MODEL

   Use trained model to make predictions on new data in a real-world application.
   
## Program:
```py
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: MIDHUN AZHAHU RAJA P
RegisterNumber: 212222240066


import pandas as pd
data = pd.read_csv("dataset/Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy

dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
```

## Output:
### INITIAL DATA SET:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393818/544d1b2e-01d3-4dbe-80aa-619e1e94e88c)
      
### DATA INFO:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393818/2c275815-3858-4208-828d-b0e5b99fb229)
      
### OPTIMIZATION OF NULL VALUES:

![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393818/d03bb3ba-7152-435f-8efa-dd4a8fe49935)
      
### ASSIGNMENT OF X AND Y VALUES:

  ![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393818/5764db49-ccb0-42df-8824-8eb1a7d5fa67)
      
 ![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393818/77fa60c0-2366-4147-809a-9f69d71d8efc)
      
### CONVERTING STRING LITERALS TO NUMERICAL VALUES USING LABEL ENCODER:

  ![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393818/7e55c68f-3b07-4e07-bde4-ab6604b6c93f)
      
### ACCURACY:

   ![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393818/839da771-9410-46be-937c-a554c01118f9)
      
### PREDICTION:

   ![image](https://github.com/MUKESHPARTHASARATHY/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393818/f615701b-1763-4398-ba44-8c65a4ae347a)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
