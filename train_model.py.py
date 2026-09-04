#!/usr/bin/env python
# coding: utf-8

# In[169]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# In[171]:


df = pd.read_csv("ai4i2020.csv")


# In[173]:


print(df.head())


# In[175]:


print(df.shape)


# In[177]:


print(df.columns)


# In[179]:


print(df.info())


# In[181]:


print(df.describe())


# In[183]:


print(df.isnull().sum())


# In[185]:


print("Duplicate rows:", df.duplicated().sum())


# In[187]:


print(df["Machine failure"].value_counts())


# In[189]:


features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


# In[191]:


X = df[features]


# In[193]:


print(X.head())


# In[195]:


y = df["Machine failure"]


# In[197]:


print(y.head())


# In[199]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[201]:


print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# In[203]:


#Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[205]:


logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
logistic_model.fit(
    X_train_scaled,
    y_train
)
logistic_pred = logistic_model.predict(X_test_scaled)
logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)
logistic_precision = precision_score(
    y_test,
    logistic_pred,
    zero_division=0
)
logistic_recall = recall_score(
    y_test,
    logistic_pred,
    zero_division=0
)
logistic_f1 = f1_score(
    y_test,
    logistic_pred,
    zero_division=0
)


# In[207]:


tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)
tree_model.fit(
    X_train,
    y_train
)
tree_pred = tree_model.predict(X_test)
tree_accuracy = accuracy_score(
    y_test,
    tree_pred
)
tree_precision = precision_score(
    y_test,
    tree_pred,
    zero_division=0
)
tree_recall = recall_score(
    y_test,
    tree_pred,
    zero_division=0
)
tree_f1 = f1_score(
    y_test,
    tree_pred,
    zero_division=0
)


# In[209]:


rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(
    X_train,
    y_train
)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)
rf_precision = precision_score(
    y_test,
    rf_pred,
    zero_division=0
)
rf_recall = recall_score(
    y_test,
    rf_pred,
    zero_division=0
)
rf_f1 = f1_score(
    y_test,
    rf_pred,
    zero_division=0
)


# In[210]:


results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        logistic_accuracy,
        tree_accuracy,
        rf_accuracy
    ],

    "Precision": [
        logistic_precision,
        tree_precision,
        rf_precision
    ],

    "Recall": [
        logistic_recall,
        tree_recall,
        rf_recall
    ],

    "F1 Score": [
        logistic_f1,
        tree_f1,
        rf_f1
    ]
})


# In[211]:


print(results)


# In[215]:


print(results.round(3))


# In[217]:


best_model = results.loc[
    results["F1 Score"].idxmax()
]
print("Best Model:")
print(best_model)


# In[219]:


print(classification_report(
    y_test,
    rf_pred,
    zero_division=0
))


# In[221]:


cm = confusion_matrix(y_test, rf_pred)

print(cm)


# In[223]:


importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

print(feature_importance)


# In[225]:


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)
print(feature_importance)


# In[227]:


new_machine = pd.DataFrame({
    "Air temperature [K]": [300],
    "Process temperature [K]": [310],
    "Rotational speed [rpm]": [1500],
    "Torque [Nm]": [45],
    "Tool wear [min]": [120]
})


# In[229]:


new_prediction = rf_model.predict(
    new_machine
)


# In[231]:


failure_probability = rf_model.predict_proba(
    new_machine
)[0][1]


# In[233]:


if failure_probability >= 0.70:
    risk = "HIGH RISK"
elif failure_probability >= 0.40:
    risk = "MEDIUM RISK"
else:
    risk = "LOW RISK"


# In[239]:


print(risk)
if new_prediction[0] == 1:
    print("Prediction: MACHINE FAILURE")
else:
    print("Prediction: NO FAILURE")


# In[241]:


import joblib

joblib.dump(rf_model, "machine_failure_model.pkl")

print("Model saved successfully!")


# In[245]:


model = joblib.load("machine_failure_model.pkl")
print(model)
print(type(model))


# In[ ]:




