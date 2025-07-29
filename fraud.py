# fraud.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("✅ Libraries imported successfully!")

# load the dataset
df = pd.read_csv("creditcard.csv")  # adjust the filename if needed
print(df.head())

df.info()
df.describe()
df.isnull().sum()

# Check class balance
df['Class'].value_counts(normalize=True)

#Plot fraud vs non-fraud counts:
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()

#Amount distributions
plt.figure(figsize=(8,4))
sns.boxplot(x='Class', y='Amount', data=df)
plt.title("Transaction Amount by Class")
plt.show()

#Time trends 
df['Hour'] = df['Time'].apply(lambda x: np.floor(x/3600) % 24)
sns.countplot(x='Hour', hue='Class', data=df)
plt.title("Transactions by Hour")
plt.show()

# Scale Amount and Time 

scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['Time_scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))

#Drop unscaled Amount, Time if desired
df = df.drop(['Time', 'Amount'], axis=1)

#Define X & y
X = df.drop('Class', axis=1)
y = df['Class']

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#SMOTE to oversample minority class

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

#Train models 
#Logistic regression
lr = LogisticRegression()
lr.fit(X_train_res, y_train_res)
y_pred_lr = lr.predict(X_test)

#Random forest
rf = RandomForestClassifier()
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test)

#XGBoost
xgb = XGBClassifier()
xgb.fit(X_train_res, y_train_res)
y_pred_xgb = xgb.predict(X_test)

#Classification report 
print(classification_report(y_test, y_pred_rf))

#confusion Matrix 
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

#ROC-AUC 
y_pred_proba = rf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

#Precision-Recall curve 
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

#Predictons probabilities 
results = X_test.copy()
results['Actual_Class'] = y_test.values
results['Predicted_Class'] = y_pred_rf
results['Fraud_Probability'] = y_pred_proba
results.to_csv("fraud_results.csv", index=False)

# RF or XGBoost 
importances = rf.feature_importances_
feat_importances = pd.Series(importances, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Top Feature Importances")
plt.show()

