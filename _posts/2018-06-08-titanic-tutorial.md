---
layout: post
title: 'Basic data-science: Predicting Titanic survivors'
date: 'Fri Jun 08 2018 02:00:00 GMT+0200 (Romance Daylight Time)'
categories: datascience
published: true
---

Let's use python to predict who will survive from the Titanic disaster!

In this project we will cover some basics of data-science and machine learning:
- clean data
- leverage the data for predictions

In short we will retrieve data from CSV files, clean the data, and train an estimator to perform binary classification.

## Step 1: get the data

Download `test.csv` and `train.csv` from [Kaggle](https://www.kaggle.com/c/titanic/data)

### Sample of `train.csv`:

````
PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley (Florence Briggs Thayer)",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
````

This is the training data, we know details about some passengers and whether
 they survived.

For instance Miss Heikkinen, a 26 y.o, 3rd class passenger did survive!

In the testing data, we don't have the "Survived" information, we have
to predict it for every passenger in the `test.csv` file.

## Step 2: clean the data

We will drop the "Name", "Ticket" and "Cabin" information as we don't care
for it.
Some values are missing, some passengers don't have an "Age", "Fare", or "Embarked"
information.

We can either drop the lines where information is missing, or we can
input some "forged" values, this is what we will do here.

{% highlight python %}
import pandas as pd 
import numpy as np
from collections import Counter

def clean_titanic(filename):
    """Clean the titanic CSV and returns the clean dataframe"""
    
    df = pd.read_csv(filename)
    # Remove unneeded info
    df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    # Fill missing age values with the mean
    age_mean = df['Age'].mean()
    df['Age'] = df['Age'].fillna(age_mean)
    # Fill missing fare values with the mean
    fare_mean = df['Fare'].mean()
    df['Fare'] = df['Fare'].fillna(fare_mean)
    # Replace strings with numbers
    df['Sex'] = df['Sex'].map({'female':0, 'male':1})
    # Fill the missing embarked values with the most common value
    embarked = Counter(df['Embarked'])
    most_common = embarked.most_common(1)[0][0]
    df['Embarked'] = df['Embarked'].fillna(most_common)
    # Replace strings with one-hot columns
    dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, dummies], axis=1)
    df.drop(['Embarked'], axis=1, inplace=True)
    
    return df

train_df = clean_titanic('train.csv')
test_df = clean_titanic('test.csv')
{% endhighlight %}

## Step 3: Extract numpy arrays from pandas dataframe

We will create 3 sets of data:
1. `X_train, Y_train` used to train our classifier
2. `X_val, Y_val` to check that our classifier can generalize from unknown data
3. `X_test` used to predict the fate of passengers whom we don't have the "Survived" info

The training labels are in the 2nd column of our training dataframe,
and we don't care for the 1st column which is the passenger ID. 

```` python
X = train_df.iloc[:,2:].values
Y = train_df['Survived'].values

X_test = test_df.iloc[:,1:].values

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3)
````

## Step 4: train an estimator

A Random forest will do the trick:

```` python
from sklearn.ensemble import RandomForestClassifier 

clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf = clf.fit(X_train, Y_train)
print(clf.score(X_val, Y_val))
````

The classifier's score on the validation data is `0.805970149254`, not 
perfect, but not bad either.

## Step 5: predict the output

We will predict from the testing data, and create file that we can upload in Kaggle.

```` python
Y_test = clf.predict(X_test)

output = pd.DataFrame(test_df['PassengerId'])
output.insert(1, "Survived", Y_test)

print(output.head())
output.to_csv(path_or_buf="titanic_predict.csv", index=False)
````

And that's it!

## Conclusion

Some things could be improved:
- Cross-Validation could help us find a better model and/or more suitable parameters.
- Training the classifier on the whole training data before applying it to the test data could improve the classifier's performance.
- And many other things...

But with some lines of python and the help of `pandas`, `scikit-learn` and `numpy` we succesfully built a machine learning  system!
