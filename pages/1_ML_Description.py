import streamlit as st

st.set_page_config(page_title="ML Model — อธิบาย", page_icon="🌲", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;600;700&family=IBM+Plex+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai', sans-serif; }
    .section {
        background: #1a1f2e; border: 1px solid #2d3748;
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1.2rem;
    }
    .section h3 { color: #63b3ed; font-size: 1.1rem; margin-bottom: 0.8rem; }
    .section p, .section li { color: #a0aec0; line-height: 1.8; }
    .section ul { padding-left: 1.2rem; }
    .tag {
        display: inline-block; padding: 0.15rem 0.6rem;
        background: #1e3a5f; color: #63b3ed;
        border-radius: 6px; font-size: 0.78rem;
        font-weight: 600; margin: 0.2rem;
    }
    .model-box {
        background: #0f1117; border: 1px solid #4a5568;
        border-radius: 8px; padding: 1rem; margin: 0.5rem 0;
    }
    .model-box h4 { color: #e2e8f0; margin: 0 0 0.3rem 0; font-size: 0.95rem; }
    .model-box p { color: #718096; margin: 0; font-size: 0.85rem; }
    .highlight { color: #68d391; font-weight: 600; }
    .ref-link { color: #63b3ed; text-decoration: none; }
    h1, h2 { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🌲 โมเดลที่ 1: ML Ensemble")
st.markdown("**GDP Growth Prediction** — ทำนายว่า GDP ของประเทศในเอเชียจะสูงกว่าค่าเฉลี่ยของตัวเองหรือไม่")

st.markdown("---")

# Dataset
st.markdown("""
<div class="section">
<h3>📂 Dataset ที่ใช้</h3>
<p><strong style="color:#e2e8f0">asia_gdp_cleaned.csv</strong> — ข้อมูล GDP Growth (%) ของประเทศในเอเชีย ปี 2013–2021</p>
<ul>
    <li><strong style="color:#e2e8f0">ที่มา:</strong> IMF World Economic Outlook Database</li>
    <li><strong style="color:#e2e8f0">จำนวนประเทศ:</strong> ครอบคลุมประเทศในภูมิภาคเอเชียแปซิฟิก</li>
    <li><strong style="color:#e2e8f0">Features:</strong> GDP % Change รายปี ตั้งแต่ปี 2013 ถึง 2020</li>
    <li><strong style="color:#e2e8f0">Target:</strong> GDP ปี 2021 สูงกว่าค่าเฉลี่ยของตัวเอง (1) หรือไม่ (0)</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Data Preparation
st.markdown("""
<div class="section">
<h3>⚙️ การเตรียมข้อมูล (Data Preprocessing)</h3>
<ul>
    <li><strong style="color:#e2e8f0">ลบ Duplicate rows</strong> — ป้องกันข้อมูลซ้ำ</li>
    <li><strong style="color:#e2e8f0">แก้ไข Inconsistent Country Name</strong> — ทำให้ชื่อประเทศสม่ำเสมอ</li>
    <li><strong style="color:#e2e8f0">จัดการ Outliers ด้วย IQR Method</strong> — แทนค่า outlier ด้วย NaN</li>
    <li><strong style="color:#e2e8f0">เติม Missing Values ด้วย Linear Interpolation</strong> — เติมตามแนวโน้มรายปี</li>
    <li><strong style="color:#e2e8f0">Feature Engineering</strong> — สร้าง mean_gdp, std_gdp, trend, last_2yr_avg</li>
    <li><strong style="color:#e2e8f0">StandardScaler</strong> — Normalize ข้อมูลก่อนเข้าโมเดล</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Algorithm Theory
st.markdown("""
<div class="section">
<h3>📖 ทฤษฎีอัลกอริทึม — Stacking Ensemble</h3>
<p>
Stacking (Stacked Generalization) เป็นเทคนิค Ensemble Learning ที่รวมหลายโมเดลเข้าด้วยกัน
โดยใช้ผลลัพธ์ของ Base Learners เป็น input ของ Meta-Learner อีกชั้นหนึ่ง
ทำให้ได้โมเดลที่มีความแม่นยำสูงกว่าการใช้โมเดลเดี่ยว
</p>
</div>
""", unsafe_allow_html=True)

# 3 Base models
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="model-box">
    <h4>🌳 Random Forest</h4>
    <p>สร้าง Decision Tree หลายต้นจาก random subset ของข้อมูล แล้ว vote ผลลัพธ์ ช่วยลด overfitting</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="model-box">
    <h4>⚡ XGBoost</h4>
    <p>Gradient Boosting ที่ปรับปรุงประสิทธิภาพด้วย regularization และ parallel processing</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="model-box">
    <h4>📈 Gradient Boosting</h4>
    <p>สร้างโมเดลแบบ sequential โดยแต่ละโมเดลพยายามแก้ error ของโมเดลก่อนหน้า</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="section" style="margin-top:1rem">
<h3>🔗 Meta-Learner: Logistic Regression</h3>
<p>รับ output จาก 3 Base Learners มาเป็น input แล้วเรียนรู้น้ำหนักที่เหมาะสมของแต่ละโมเดล
เพื่อทำการ predict ขั้นสุดท้าย</p>
</div>
""", unsafe_allow_html=True)

# Steps
st.markdown("""
<div class="section">
<h3>🛠️ ขั้นตอนการพัฒนาโมเดล</h3>
<ul>
    <li>แบ่งข้อมูล Train 80% / Test 20% ด้วย stratified split</li>
    <li>ใช้ 5-Fold Cross-Validation ประเมิน Base Learners แต่ละตัว</li>
    <li>สร้าง StackingClassifier ด้วย <code style="background:#0f1117;padding:2px 6px;border-radius:4px;color:#68d391">passthrough=True</code> เพื่อส่ง original features ด้วย</li>
    <li>ประเมินผลด้วย Accuracy, Precision, Recall, F1-Score</li>
    <li>วิเคราะห์ Feature Importance จาก Random Forest</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Images
st.markdown("### 📊 ผลการประเมินโมเดล")
col1, col2 = st.columns(2)
with col1:
    try:
        st.image("models/images/confusion_matrix_ml.png", caption="Confusion Matrix", use_column_width=True)
    except:
        st.info("ยังไม่พบรูป — รัน train_models_v2.py ก่อน")
with col2:
    try:
        st.image("models/images/feature_importance_ml.png", caption="Feature Importance", use_column_width=True)
    except:
        st.info("ยังไม่พบรูป — รัน train_models_v2.py ก่อน")

# References
st.markdown("""
<div class="section">
<h3>📚 แหล่งอ้างอิง</h3>
<ul>
    <li><a class="ref-link" href="https://scikit-learn.org/stable/modules/ensemble.html" target="_blank">Scikit-learn Documentation — Ensemble Methods</a></li>
    <li><a class="ref-link" href="https://xgboost.readthedocs.io/" target="_blank">XGBoost Documentation</a></li>
    <li><a class="ref-link" href="https://www.imf.org/en/Publications/WEO" target="_blank">IMF World Economic Outlook Database</a></li>
    <li>Wolpert, D.H. (1992). Stacked Generalization. Neural Networks, 5(2), 241–259.</li>
</ul>
</div>
""", unsafe_allow_html=True)
