# %%
# # 1. Customer Churn Prediction
# Industry: Telecom, SaaS, Retail
# Problem: Predict whether a customer will stop using a service or product. Companies can use this information to take
#  preventative actions.
# Dataset: Telco Customer Churn
# Key Algorithms: Logistic Regression, Random Forest, Gradient Boosting
# Why It’s Relevant: Customer retention is a huge priority for subscription-based businesses. Reducing churn directly 
# affects revenue.

# %%
import pandas as pd 
import numpy as nu 


# %%
df = pd.read_csv(r"C:\Users\vishal\Downloads\Telco-Customer-Churn.csv")
df

# %%
df.isnull().sum()

# %%
df.drop(columns = ["customerID"], inplace=True)

# %%
df.duplicated().sum()

# %%
df.drop_duplicates(inplace=True)

# %%
df.duplicated().sum()

# %%
df["Partner"] = df["Partner"].map({"Yes":0,"No":1})

# %%
df["gender"] = df["gender"].map({"Female":0,"Male":1})

# %%
df["Dependents"].value_counts()

# %%
df["Dependents"] = df["Dependents"].map({"No":0,"Yes":1})

# %%
df["PhoneService"] = df["PhoneService"].map({"No":0,"Yes":1})

# %%
df["MultipleLines"] = df["MultipleLines"].map({"No":0,"Yes":1,"No phone service":2})

# %%
df["InternetService"] = df["InternetService"].map({"DSL":0,"Fiber optic":1,"No":2})

# %%
df["OnlineSecurity"] = df["OnlineSecurity"].map({"No":0,"Yes":1,"No internet service":2})

# %%
df["OnlineBackup"] = df["OnlineBackup"].map({"No":0,"Yes":1,"No internet service":2})

# %%
df["DeviceProtection"] = df["DeviceProtection"].map({"No":0,"Yes":1,"No internet service":2})

# %%
df["TechSupport"] = df["TechSupport"].map({"No":0,"Yes":1,"No internet service":2})

# %%
df["StreamingTV"] = df["StreamingTV"].map({"No":0,"Yes":1,"No internet service":2})

# %%
df["StreamingMovies"] = df["StreamingMovies"].map({"No":0,"Yes":1,"No internet service":2})

# %%
df["Contract"] = df["Contract"].map({"Month-to-month":0,"One year":1,"Two year":2})

# %%
df["PaperlessBilling"] = df["PaperlessBilling"].map({"No":0,"Yes":1})

# %%
df['PaymentMethod'] = df['PaymentMethod'].map({"Electronic check":0,"Mailed check":1,"Bank transfer (automatic)":2,"Credit card (automatic)":3})


# %%
df["Churn"] = df["Churn"].map({"No":0,"Yes":1})

# %%
df["tenure"].value_counts()

# %%
df['tenure'] = pd.qcut(df['tenure'], q=4, labels=["Low", "Medium", "High", "Very High"])


# %%
df["tenure"] = df["tenure"].map({"Low":0,"Very High":1,"High":2,"Medium":3})

# %%
#['MonthlyCharges'] = pd.qcut(df['MonthlyCharges'], q=4, labels=["Low", "Medium", "High", "Very High"])
df["MonthlyCharges"] = df["MonthlyCharges"].map({"Low":0,"Very High":1,"High":2,"Medium":3})

# %%
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


# %%
df['TotalCharges'] = pd.qcut(df['TotalCharges'], q=4, labels=["Low", "Medium", "High", "Very High"])


# %%
df["TotalCharges"] = df["TotalCharges"].map({"Low":0,"Very High":1,"High":2,"Medium":3})

# %%
df["TotalCharges"].value_counts()

# %%
df

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# %%
x = df.drop(columns=["Churn"])

# %%
y = df["Churn"]

# %%
#X_train, X_test, Y_train, Y_test = train_test_split(x,y)

# %%
#clf = RandomForestClassifier()

# %%
#model = clf.fit(X_train,Y_train)

# %%
#Y_pred = model.predict(X_test)

# %%
#accuracy_score(Y_test,Y_pred)

# %%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# %%
from sklearn.linear_model import LogisticRegression
logreg_model = LogisticRegression()

# %%
logreg_model.fit(X_train, y_train)

# %%
y_pred = logreg_model.predict(X_test)


# %%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)

# %%
accuracy_score(Y_test,Y_pred)

# %%


# %%


# %%


# %%


# %%



