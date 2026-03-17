import streamlit as st

st.set_page_config(
    page_title="Asia Economy AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;600;700&family=IBM+Plex+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans Thai', sans-serif;
    }
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }

    .hero {
        background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0f1117 100%);
        border: 1px solid #2d3748;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 50% 50%, rgba(99,179,237,0.05) 0%, transparent 60%);
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
    }
    .hero h1 span { color: #63b3ed; }
    .hero p {
        color: #718096;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    .card-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-top: 1.5rem;
    }
    .card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.5rem;
        transition: border-color 0.2s;
    }
    .card:hover { border-color: #63b3ed; }
    .card-icon { font-size: 2rem; margin-bottom: 0.5rem; }
    .card h3 { color: #e2e8f0; font-size: 1rem; font-weight: 600; margin: 0.3rem 0; }
    .card p { color: #718096; font-size: 0.85rem; margin: 0; }

    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.3rem;
    }
    .badge-blue { background: #1e3a5f; color: #63b3ed; }
    .badge-green { background: #1a3a2a; color: #68d391; }

    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #2d3748;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="card-icon">🌏</div>
    <h1>Asia Economy <span>AI Project</span></h1>
    <p>วิเคราะห์เศรษฐกิจเอเชียด้วย Machine Learning และ Neural Network</p>
    <br>
    <span class="badge badge-blue">Project IS 2568</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card-grid">
    <div class="card">
        <div class="card-icon">🌲</div>
        <h3>ML Ensemble Model</h3>
        <p>ทำนาย GDP Growth ขึ้น/ลง ด้วย Stacking Classifier<br>Random Forest + XGBoost + Gradient Boosting</p>
    </div>
    <div class="card">
        <div class="card-icon">🧠</div>
        <h3>Neural Network (LSTM)</h3>
        <p>วิเคราะห์ Sentiment ของข่าวเศรษฐกิจ<br>Bidirectional LSTM สำหรับ Text Classification</p>
    </div>
    <div class="card">
        <div class="card-icon">📊</div>
        <h3>Dataset 1: Asia GDP</h3>
        <p>ข้อมูล GDP Growth (%) ของประเทศในเอเชีย<br>ปี 2013–2021 (Structured Data)</p>
    </div>
    <div class="card">
        <div class="card-icon">📰</div>
        <h3>Dataset 2: Asia News</h3>
        <p>ข่าวเศรษฐกิจเอเชียพร้อม Sentiment Label<br>(Unstructured / Text Data)</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.info("👈 เลือกหน้าจาก **Sidebar** ด้านซ้ายเพื่อดูรายละเอียดหรือทดสอบโมเดล")
