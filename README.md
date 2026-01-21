# 📌 Customer Churn Prediction — XGBoost + Streamlit

This project predicts customer churn for a telecom subscription platform using machine learning.  
It supports batch CSV uploads and returns churn probabilities and churn labels for each customer.

## 🚀 Key Features

- XGBoost model optimized using **GridSearchCV**
- Achieved **~84% ROC-AUC** on ~7K customer dataset
- Used **Pipeline + ColumnTransformer** for preprocessing
- **Batch CSV inference** (no manual UI form needed)
- Streamlit interface for easy usage
- Outputs downloadable prediction CSV
- Threshold-based churn classification

## 🧠 Modeling Approach

Training steps included:

- Data cleaning (`TotalCharges`)
- Dropping `customerID`
- OneHotEncoding categorical features
- Numeric passthrough
- Pipeline + GridSearchCV
- Metric: ROC-AUC

## 📂 Input Format (CSV)

```
customerID
gender
SeniorCitizen
Partner
Dependents
tenure
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
MonthlyCharges
TotalCharges
```

## 🧹 Runtime Preprocessing

- Blank `TotalCharges` → NaN → **Drop row**
- Remaining columns passed directly to Pipeline
- Pipeline handles encoding

## 📈 Output Format

```
customerID
churn_prob   (0.0–1.0)
churn_label  (0 or 1)
```

Threshold used: **0.35**

## 🖥 Tech Stack

| Component | Library |
|---|---|
| Model | XGBoost |
| Encoding | ColumnTransformer + OneHotEncoder |
| Tuning | GridSearchCV |
| Deployment | Streamlit |
| Serialization | Joblib |
| Data Handling | Pandas / NumPy |

## 🔧 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Repository Structure

```
├── app.py                   # Streamlit App
├── xgb_churn_model.pkl      # Trained Pipeline + Model
├── sample.csv               # Example Input CSV
├── README.md
└── requirements.txt
```

## 📊 Performance Summary

| Metric | Score |
|---|---|
| ROC-AUC | ~0.84 |
| Accuracy | ~0.80 |
| Precision (Churn) | ~0.66 |
| Recall (Churn) | ~0.50 |

## 🌱 Use Cases

- Telecom churn prediction
- Subscription-based platforms
- Customer retention analytics
- Batch scoring systems

## 🏁 Future Improvements

- SHAP feature importance
- Cost-based thresholding
- REST API (FastAPI)
- Visualization dashboard

## 📜 License

Open for non-commercial educational use.
