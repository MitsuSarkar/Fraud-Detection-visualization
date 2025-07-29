# Fraud-Detection-visualization

# 🕵️‍♂️ Credit Card Fraud Detection with Machine Learning & Power BI

This project combines **machine learning in Python** and **data visualization in Power BI** to detect and explore fraudulent credit card transactions. It demonstrates end-to-end data science workflows including modeling, evaluation, and interactive dashboarding.

---

## 📌 Project Goals

- Build a fraud detection model using Python and machine learning.
- Handle class imbalance using SMOTE.
- Evaluate model performance using metrics like precision, recall, and F1-score.
- Visualize fraud patterns and model results with a Power BI dashboard.
- Communicate insights clearly to both technical and non-technical stakeholders.

---

## 📂 Dataset

- 📄 Source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- 📊 284,807 transactions
- 🧪 Highly imbalanced:
  - ~0.17% fraudulent
- ✅ Features:
  - `V1` to `V28`: anonymized PCA components
  - `Amount`, `Time`
  - `Class`: 0 = Legit, 1 = Fraud

---

## 🧪 Tools Used

| Category      | Tools/Libraries                          |
|---------------|-------------------------------------------|
| Programming   | Python (pandas, scikit-learn, xgboost)   |
| Modeling      | Logistic Regression, Random Forest, XGBoost |
| Imbalance     | SMOTE (imbalanced-learn)                |
| Visualization | matplotlib, seaborn                      |
| Dashboarding  | Power BI                                 |

---

## 🔍 Python Workflow (Jupyter Notebook)

### ✔️ Step 1: Data Exploration (EDA)
- Checked nulls, distributions, class imbalance
- Explored fraud patterns over `Amount`, `Time`, `Hour`

### ✔️ Step 2: Preprocessing
- Scaled `Amount` and `Time`
- Created `Hour` column from `Time`
- Split data into training and test sets

### ✔️ Step 3: Handling Imbalance
- Applied **SMOTE** to oversample minority class

### ✔️ Step 4: Model Training
- Trained:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Used cross-validation and baseline tuning

### ✔️ Step 5: Evaluation
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Plotted ROC and Precision-Recall curves
- Extracted predicted probabilities

### ✔️ Step 6: Export Results
- Created `fraud_results.csv` with:
  - `Actual_Class`, `Predicted_Class`, `Fraud_Probability`, `Hour`, `Amount`, etc.

---

## 📊 Power BI Dashboard

Built a professional dashboard in Power BI to present findings:

### 🔹 KPIs
- Total Transactions
- Actual Frauds
- Predicted Frauds
- Fraud Rate (%)

### 🔹 Visuals
- **Line chart**: Fraud activity & F1 Score by Hour
- **Pie charts**: Fraud distribution by actual/predicted class
- **Matrix**: Confusion Matrix
- **Histogram**: Fraud probability buckets
- **Boxplot / Bar**: Fraud by amount & time

### 🔹 Interactivity
- Slicers: Actual/Predicted class, probability buckets, hours
- Dynamic filtering across visuals

📷 
<img width="1342" height="781" alt="Screenshot (34)" src="https://github.com/user-attachments/assets/769a3210-d0c0-46bd-b821-bea0c41da1db" />



## ▶️ How to Run This Project

### 🧪 Python Model
1. Clone the repo
2. Open `fraud_detection.ipynb` or `fraud.py`
3. Install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn

yaml
Copy
Edit
4. Run the notebook to train models and generate `fraud_results.csv`

### 📊 Power BI Dashboard
1. Open `fraud_dashboard.pbix` in Power BI Desktop
2. Load `fraud_results.csv` into the model
3. Explore the interactive dashboard


## ✅ Project Outcomes

- Achieved >95% precision and recall using Random Forest
- Built interpretable & stakeholder-ready visuals in Power BI
- Combined data science and BI skills to address real-world fraud detection

---

## 🚀 Future Improvements

- Hyperparameter tuning with GridSearchCV
- Add neural network (e.g., simple MLP)
- Automate data refresh for live dashboards
- Deploy as a Streamlit app

---

## 👤 Author

**Sumit Sarkar**  
📍 London, UK  
🎓 MSc Finance  
📧 [sumitsarkar2222@gmail.com](mailto:sumitsarkar2222@gmail.com)  
🔗 [linkedin.com/in/mitsusarkar](https://linkedin.com/in/mitsusarkar)
