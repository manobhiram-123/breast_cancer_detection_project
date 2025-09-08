import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("breast_cancer_data.csv.csv")

df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE


X=df[['texture_mean', 'perimeter_mean', 'concavity_mean',
       'concave points_mean', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'concavity_worst',
       'concave points_worst']]
y=df['diagnosis']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


model=RandomForestClassifier( class_weight='balanced', random_state=42)





model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print("accuracy_score :",accuracy_score(y_test,y_pred))
print("confusion_matrix :")
print(confusion_matrix(y_test,y_pred))
print("classification_report :")
print(classification_report(y_test,y_pred))





data = pd.DataFrame({
    'texture_mean': [18.5],
    'perimeter_mean': [125.0],
    'concavity_mean': [0.25],
    'concave points_mean': [0.12],
    'radius_worst': [24.0],
    'texture_worst': [18.0],
    'perimeter_worst': [180.0],
    'area_worst': [2000],
    'concavity_worst': [0.35],
    'concave points_worst': [0.18]
})

new=sc.transform(data)





pred_class = model.predict(data)[0]  


pred_proba = model.predict_proba(data)[0]  


pred_class_prob_percent = pred_proba[pred_class] * 100


print(f"Predicted class: {pred_class} ({'Benign' if pred_class==0 else 'Malignant'})")
print(f"Probability of predicted class: {pred_class_prob_percent:.2f}%")



import joblib

joblib.dump(model, 'breast_cancer_model.pkl')
joblib.dump(sc, 'scaler.pkl')

