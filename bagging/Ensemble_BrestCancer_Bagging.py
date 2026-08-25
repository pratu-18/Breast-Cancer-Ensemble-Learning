#----------------------------------
#Bagging  Program which works multiple model paralary
#-------------------------------------

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

#-----------------------------------------------------------------
#Step 1:load the dataset
#----------------------------------------------------------------


Border="-"*80
df=pd.read_csv("breast_cancer.csv")
print("Shape of dataset :",df.shape)
print("First 5 records :")
print(df.head())
print(Border)

#--------------------------------------------------------------
#Step 2: Separete features and label
#-------------------------------------------------------------

X=df.drop("target",axis=1)
Y=df["target"]

print("X shape are =",X.shape)
print("Y shape are =",Y.shape)
print(Border)
#--------------------------------------------------------------
#Step 3: Split dataset for training and testing
#-------------------------------------------------------------

X_train,X_test,Y_train,Y_test=train_test_split(X,
                                               Y,
                                               test_size=0.2,
                                               random_state=42)


print(Border)
#--------------------------------------------------------------
#Step 4: Scale the features
#-------------------------------------------------------------

scalar=StandardScaler()
X_train=scalar.fit_transform(X_train)
X_test=scalar.fit_transform(X_test)
print(Border)

#--------------------------------------------------------------
#Step 5.1: Create the base model
#-------------------------------------------------------------

base_model=DecisionTreeClassifier(random_state=42)



#--------------------------------------------------------------
#Step 5.2: Create the bagging model
#-------------------------------------------------------------

model=BaggingClassifier(
    estimator=base_model,
    n_estimators=10  ,                    #base_mdel che replica 10 banav means it creates multiple decision tree
    random_state=42

)

#--------------------------------------------------------------
#Step 6: Train the model
#-------------------------------------------------------------

model=model.fit(X_train,Y_train)

#--------------------------------------------------------------
#Step 7: Test the model
#-------------------------------------------------------------

Y_pred=model.predict(X_test)


#--------------------------------------------------------------
#Step 8: Evalute  the model
#-------------------------------------------------------------

print("Accuracy  :",accuracy_score(Y_test,Y_pred))
print("Confusion matrix :")
print(confusion_matrix(Y_test,Y_pred))

