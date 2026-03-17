import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="ทดสอบ ML Model", page_icon="🌲", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai', sans-serif; }
    .result-up {
        background: linear-gradient(135deg, #1a3a2a, #22543d);
        border: 1px solid #68d391; border-radius: 12px;
        padding: 2rem; text-align: center;
    }
    .result-down {
        background: linear-gradient(135deg, #3a1a1a, #742a2a);
        border: 1px solid #fc8181; border-radius: 12px;
        padding: 2rem; text-align: center;
    }
    .result-up h2, .result-down h2 { font-size: 2rem; margin: 0.3rem 0; }
    .result-up p { color: #68d391; margin: 0; }
    .result-down p { color: #fc8181; margin: 0; }
    .input-section {
        background: #1a1f2e; border: 1px solid #2d3748;
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
    }
    .input-section h3 { color: #63b3ed; margin-bottom: 1rem; font-size: 1rem; }
    h1 { color: #e2e8f0; }
    .stSlider label { color: #a0aec0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🌲 ทดสอบ ML Ensemble Model")
st.markdown("กรอกข้อมูล GDP Growth (%) ของประเทศ แล้วโมเดลจะทำนายว่า GDP ปี 2021 จะ **สูงกว่าค่าเฉลี่ย** หรือ **ต่ำกว่าค่าเฉลี่ย**")
st.markdown("---")

# Load model
@st.cache_resource
def load_ml_model():
    model = joblib.load("models/ensemble_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    cols = joblib.load("models/feature_columns.pkl")
    return model, scaler, cols

try:
    model, scaler, feature_cols = load_ml_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ โหลดโมเดลไม่ได้: {e}")
    model_loaded = False

if model_loaded:
    st.markdown('<div class="input-section"><h3>📥 กรอกข้อมูล GDP Growth (%) รายปี</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    years = list(range(2013, 2021))
    gdp_values = {}

    # Default ค่าตัวอย่าง (ประมาณค่าทั่วไปของประเทศเอเชีย)
    defaults = {2013: 5.0, 2014: 4.5, 2015: 4.0, 2016: 3.8, 2017: 4.2, 2018: 4.0, 2019: 3.5, 2020: -2.0}

    for i, year in enumerate(years):
        col = col1 if i < 4 else col2
        with col:
            gdp_values[str(year)] = st.slider(
                f"GDP {year} (%)",
                min_value=-15.0, max_value=20.0,
                value=float(defaults[year]), step=0.1,
                key=f"gdp_{year}"
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # สร้าง feature vector
    year_cols_list = [str(y) for y in range(2013, 2021)]
    vals = [gdp_values[y] for y in year_cols_list]

    mean_gdp     = np.mean(vals)
    std_gdp      = np.std(vals)
    trend        = vals[-1] - vals[0]
    last_2yr_avg = (vals[-2] + vals[-1]) / 2

    feature_dict = {y: [gdp_values[y]] for y in year_cols_list}
    feature_dict["mean_gdp"]     = [mean_gdp]
    feature_dict["std_gdp"]      = [std_gdp]
    feature_dict["trend"]        = [trend]
    feature_dict["last_2yr_avg"] = [last_2yr_avg]

    X_input = pd.DataFrame(feature_dict)[feature_cols]

    # แสดง summary
    st.markdown("### 📊 สรุปข้อมูลที่กรอก")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("ค่าเฉลี่ย GDP", f"{mean_gdp:.2f}%")
    mc2.metric("ความผันผวน (Std)", f"{std_gdp:.2f}%")
    mc3.metric("แนวโน้มรวม (Trend)", f"{trend:.2f}%")
    mc4.metric("เฉลี่ย 2 ปีล่าสุด", f"{last_2yr_avg:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮 ทำนาย GDP ปี 2021", use_container_width=True, type="primary"):
        X_scaled = scaler.transform(X_input)
        prediction = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]

        if prediction == 1:
            st.markdown(f"""
            <div class="result-up">
                <h2>📈 GDP สูงกว่าค่าเฉลี่ย</h2>
                <p>โมเดลทำนายว่า GDP ปี 2021 จะ <strong>สูงกว่าค่าเฉลี่ย</strong> ของประเทศนี้</p>
                <p style="margin-top:0.5rem;color:#a0aec0">ความมั่นใจ: {proba[1]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-down">
                <h2>📉 GDP ต่ำกว่าค่าเฉลี่ย</h2>
                <p>โมเดลทำนายว่า GDP ปี 2021 จะ <strong>ต่ำกว่าค่าเฉลี่ย</strong> ของประเทศนี้</p>
                <p style="margin-top:0.5rem;color:#a0aec0">ความมั่นใจ: {proba[0]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Probability ของแต่ละ class:**")
        prob_df = pd.DataFrame({
            "Class": ["GDP ต่ำกว่าค่าเฉลี่ย (0)", "GDP สูงกว่าค่าเฉลี่ย (1)"],
            "Probability": [f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"]
        })
        st.table(prob_df)

    # ตัวอย่างข้อมูลจริง
    with st.expander("💡 ตัวอย่างข้อมูลจริงจาก Dataset"):
        try:
            df = pd.read_csv("data/asia_gdp_cleaned.csv")
            st.dataframe(df.head(10), use_container_width=True)
        except:
            st.info("ไม่พบไฟล์ data/asia_gdp_cleaned.csv")
