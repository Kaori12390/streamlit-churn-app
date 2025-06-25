import streamlit as st
import pandas as pd
import joblib

# CSS hiện đại: đen + xanh neon
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f0f0f;
        color: #00ffcc;
        font-family: 'Segoe UI', sans-serif;
        padding: 20px;
    }

    h1, h2, h3 {
        color: #00ffcc;
    }

    .stButton > button {
        background-color: #00ffcc;
        color: black;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }

    .stNumberInput > div > input {
        background-color: #1a1a1a;
        color: #00ffcc;
        border: 1px solid #00ffcc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model và danh sách feature
model = joblib.load("model/best_lgbm_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

# Tiêu đề căn giữa
st.markdown("<h1 style='text-align: center;'>🔮 Dự đoán khách hàng rời bỏ ngân hàng</h1>", unsafe_allow_html=True)

# Giao diện nhập liệu
st.subheader("📥 Nhập thông tin khách hàng:")

user_input = {}
cols = st.columns(2)  # chia làm 2 cột

for idx, col in enumerate(feature_columns):
    with cols[idx % 2]:  # luân phiên trái-phải
        user_input[col] = st.number_input(f"{col}", step=1.0)

# Dự đoán khi bấm nút
if st.button("📊 Dự đoán"):
    input_df = pd.DataFrame([user_input])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Khách hàng có khả năng rời bỏ!")
    else:
        st.success("✅ Khách hàng sẽ ở lại.")
