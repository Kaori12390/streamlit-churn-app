import streamlit as st
import pandas as pd
import joblib

# Load model và danh sách feature
model = joblib.load("model/best_lgbm_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

st.title("🧠 Dự đoán khách hàng rời bỏ ngân hàng")

# Giao diện nhập liệu
user_input = {}
st.subheader("Nhập thông tin khách hàng:")

for col in feature_columns:
    user_input[col] = st.number_input(f"{col}", step=1.0)

if st.button("📊 Dự đoán"):
    input_df = pd.DataFrame([user_input])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    prediction = model.predict(input_df)[0]
    
    if prediction == 1:
        st.error("⚠️ Khách hàng có khả năng rời bỏ!")
    else:
        st.success("✅ Khách hàng sẽ ở lại.")