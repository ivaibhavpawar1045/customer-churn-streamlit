import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

# --------------------------------------------------
# MUST be the first Streamlit command
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# --------------------------------------------------
# App Header
# --------------------------------------------------
st.title("Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")

# --------------------------------------------------
# Load Dataset Safely
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/Telco-Customer-Churn.csv")
    if df.empty:
        st.error("Dataset is empty. Please check the CSV file.")
        st.stop()
    return df

df = load_data()

# --------------------------------------------------
# Data Preprocessing
# --------------------------------------------------
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

cat_cols = df.select_dtypes(include="object").columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop(["Churn", "customerID"], axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# Model Training
# --------------------------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --------------------------------------------------
# Model Performance
# --------------------------------------------------
st.subheader("Model Performance")

y_pred = model.predict(X_test)

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 2))
c2.metric("Recall", round(recall_score(y_test, y_pred), 2))
c3.metric("ROC-AUC", round(roc_auc_score(y_test, y_pred), 2))

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.subheader("Predict Customer Churn")

user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(
        label=col,
        min_value=float(X[col].min()),
        max_value=float(X[col].max()),
        value=float(X[col].mean())
    )

input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is not likely to churn")
