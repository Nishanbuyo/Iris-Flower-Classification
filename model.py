# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import model_selection

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split, KFold, cross_val_score



#Loading data
iris=pd.read_csv("iris_data.csv")

print(iris.head())
print(iris.shape)
print(iris.describe())
print(iris.groupby("class").size())

iris.hist()
plt.show()
scatter_matrix(iris)
plt.show()

X=iris.drop("class", axis=1).values
Y=iris["class"]
# print(X)
# print(Y)
# le=preprocessing.LabelEncoder()
# le.fit(['Iris-setosa','Iris-versicolor', 'Iris-virginica'])
# Y[:]=le.transform(Y[:])

#Train test split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2, random_state=1)

#Spot Check Algorithm
models=[]
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('SVM',SVC(kernel='rbf', gamma='auto')))
models.append(('GNB', GaussianNB()))
# print(X_train, Y_train)
results=[]
names=[]
for name, model in models:
    kfold=KFold(n_splits=10, shuffle=True, random_state=1)
    kf_result=cross_val_score(model, X_train, Y_train,cv=kfold, scoring='accuracy')
    results.append(kf_result)
    names.append(name)
    print('%s: %f (%f)' % (name, kf_result.mean(), kf_result.std()))

# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

#Prediction
model=SVC(gamma='auto', kernel='rbf')
model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)
print(Y_pred)

#Evaluate Prediction
print(accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

# Saving model
from joblib import load, dump
dump(model, "iris_model.joblib")