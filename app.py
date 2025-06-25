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

# ===== Danh sách tỉnh/thành phố với mã =====
province_options = {
    "Hà Nội": 1, "Hà Giang": 2, "Cao Bằng": 4, "Bắc Kạn": 6, "Tuyên Quang": 8, "Lào Cai": 10,
    "Điện Biên": 11, "Lai Châu": 12, "Sơn La": 14, "Yên Bái": 15, "Hòa Bình": 17, "Thái Nguyên": 19,
    "Lạng Sơn": 20, "Quảng Ninh": 22, "Bắc Giang": 24, "Phú Thọ": 25, "Vĩnh Phúc": 26, "Bắc Ninh": 27,
    "Hải Dương": 30, "Hải Phòng": 31, "Hưng Yên": 33, "Thái Bình": 34, "Hà Nam": 35, "Nam Định": 36,
    "Ninh Bình": 37, "Thanh Hóa": 38, "Nghệ An": 40, "Hà Tĩnh": 42, "Quảng Bình": 44, "Quảng Trị": 45,
    "Thừa Thiên Huế": 46, "Đà Nẵng": 48, "Quảng Nam": 49, "Quảng Ngãi": 51, "Bình Định": 52,
    "Phú Yên": 54, "Khánh Hòa": 56, "Ninh Thuận": 58, "Bình Thuận": 60, "Đồng Nai": 61, "Kon Tum": 62,
    "Gia Lai": 64, "Đắk Lắk": 66, "Đắk Nông": 67, "Lâm Đồng": 68, "Kiên Giang": 69, "Bình Phước": 70,
    "Hậu Giang": 71, "Tây Ninh": 72, "Cần Thơ": 73, "Bình Dương": 74, "Bà Rịa – Vũng Tàu": 75,
    "Sóc Trăng": 76, "Bạc Liêu": 77, "Cà Mau": 78, "Hồ Chí Minh": 79, "Long An": 80,
    "Tiền Giang": 81, "Bến Tre": 82, "Trà Vinh": 84, "Vĩnh Long": 86, "An Giang": 67, "Đồng Tháp": 66
}

# ===== Giao diện chính =====
st.markdown("<h1 style='text-align: center;'>🔮 Dự đoán khách hàng rời bỏ ngân hàng</h1>", unsafe_allow_html=True)
st.subheader("📥 Nhập thông tin khách hàng:")

# ===== Trường hiển thị (dropdown hoặc số) =====
visible_fields = {
    'cust_age': {
        'label': '👤 Tuổi của khách hàng là bao nhiêu?',
        'type': 'number'
    },
    'income': {
        'label': '💰 Thu nhập hàng tháng (triệu VND)',
        'type': 'number'
    },
    'gender': {
        'label': '⚧️ Giới tính',
        'type': 'select',
        'options': ['Nữ', 'Nam']
    },
    'cust_tenure': {
        'label': '📅 Khách hàng đã sử dụng dịch vụ bao lâu (năm)?',
        'type': 'number'
    },
    'resid_province': {
        'label': '🏠 Tỉnh cư trú',
        'type': 'select',
        'options': list(province_options.keys())
    }
}

# ===== Giao diện nhập liệu chia cột =====
user_input = {}
cols = st.columns(2)
for idx, (col, config) in enumerate(visible_fields.items()):
    with cols[idx % 2]:
        if config['type'] == 'number':
            user_input[col] = st.number_input(config['label'], step=1.0)
        elif config['type'] == 'select':
            user_input[col] = st.selectbox(config['label'], config['options'])

# ===== Mã hóa lại gender & province =====
if 'gender' in user_input:
    gender_map = {'Nữ': 0, 'Nam': 1}
    user_input['gender'] = gender_map[user_input['gender']]

if 'resid_province' in user_input:
    user_input['resid_province'] = province_options[user_input['resid_province']]

# ===== Điền các trường còn thiếu = 0 =====
for col in feature_columns:
    if col not in user_input:
        user_input[col] = 0

# ===== Dự đoán kết quả =====
if st.button("📊 Dự đoán"):
    input_df = pd.DataFrame([user_input])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Khách hàng có khả năng rời bỏ!")
    else:
        st.success("✅ Khách hàng sẽ ở lại.")
