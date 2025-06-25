import streamlit as st
import pandas as pd
import joblib

# ===== CSS phong cách ngân hàng truyền thống =====
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
        color: #004080;
        font-family: 'Segoe UI', sans-serif;
        padding: 30px;
    }

    h1, h2, h3 {
        color: #004080;
    }

    .stButton > button {
        background-color: #004080;
        color: white;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
    }

    .stNumberInput > div > input {
        background-color: white;
        color: #004080;
        border: 1px solid #004080;
    }

    .stSuccess {
        background-color: #e6f2ff;
        border-left: 5px solid #004080;
    }

    .stError {
        background-color: #fff4e6;
        border-left: 5px solid #cc3300;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===== Load mô hình và danh sách feature =====
model = joblib.load("model/best_lgbm_model.pkl")
feature_columns = joblib.load("model/feature_columns.pkl")

# ===== Tiêu đề chính =====
st.markdown("<h1 style='text-align: center;'>🔮 Dự đoán khách hàng rời bỏ ngân hàng</h1>", unsafe_allow_html=True)
st.subheader("📥 Nhập thông tin khách hàng:")

# ===== Danh sách trường cần hiển thị + câu hỏi thân thiện =====
visible_fields = {
    'cust_age': '👤 Tuổi của khách hàng là bao nhiêu?',
    'income': '💰 Thu nhập hàng tháng (triệu VND)',
    'gender': '⚧️ Giới tính (0: Nữ, 1: Nam)',
    'cust_tenure': '📅 Khách hàng đã sử dụng dịch vụ bao lâu (năm)?',
    'resid_province': '🏠 Mã tỉnh cư trú của khách hàng'
}

# ===== Giao diện nhập liệu chia 2 cột =====
user_input = {}
cols = st.columns(2)
for idx, (col, label) in enumerate(visible_fields.items()):
    with cols[idx % 2]:
        user_input[col] = st.number_input(label, step=1.0)

# ===== Điền giá trị 0 cho các trường ẩn không hiển thị =====
for col in feature_columns:
    if col not in user_input:
        user_input[col] = 0

# ===== Dự đoán khi người dùng bấm nút =====
if st.button("📊 Dự đoán"):
    input_df = pd.DataFrame([user_input])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Khách hàng có khả năng rời bỏ!")
    else:
        st.success("✅ Khách hàng sẽ ở lại.")
