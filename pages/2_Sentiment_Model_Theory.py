import streamlit as st

st.set_page_config(page_title="Sentiment Model — Theory", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;600;700&family=IBM+Plex+Mono&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai', sans-serif; }
    .section {
        background: #1a1f2e; border: 1px solid #2d3748;
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1.2rem;
    }
    .section h3 { color: #9f7aea; font-size: 1.1rem; margin-bottom: 0.8rem; }
    .section p, .section li { color: #a0aec0; line-height: 1.8; }
    .section ul { padding-left: 1.2rem; }
    .layer-box {
        background: #0f1117; border-left: 3px solid #9f7aea;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.4rem 0;
    }
    .layer-box h4 { color: #e2e8f0; margin: 0 0 0.2rem 0; font-size: 0.9rem; }
    .layer-box p { color: #718096; margin: 0; font-size: 0.82rem; }
    .ref-link { color: #9f7aea; text-decoration: none; }
    h1, h2 { color: #e2e8f0; }
    code { background:#0f1117; padding:2px 6px; border-radius:4px; color:#b794f4; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🧠 โมเดลที่ 2: Neural Network (LSTM)")
st.markdown("**Sentiment Classification** — วิเคราะห์ความรู้สึกของข่าวเศรษฐกิจว่าเป็น Positive / Neutral / Negative")

st.markdown("---")

# Dataset
st.markdown("""
<div class="section">
<h3>📂 Dataset ที่ใช้</h3>
<p><strong style="color:#e2e8f0">asia_news_cleaned.csv</strong> — ข่าวเศรษฐกิจเอเชียพร้อม Sentiment Label</p>
<ul>
    <li><strong style="color:#e2e8f0">ที่มา:</strong> Reuters, Bloomberg, Nikkei Asia (สร้างจากข้อมูลจริง)</li>
    <li><strong style="color:#e2e8f0">Features:</strong> headline (ข้อความข่าว), country, year, gdp_change</li>
    <li><strong style="color:#e2e8f0">Target:</strong> sentiment_label — Negative (0), Neutral (1), Positive (2)</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Data Preparation
st.markdown("""
<div class="section">
<h3>⚙️ การเตรียมข้อมูล (Data Preprocessing)</h3>
<ul>
    <li><strong style="color:#e2e8f0">Text Cleaning</strong> — ลบ special characters, แก้ typo, ลบ extra spaces</li>
    <li><strong style="color:#e2e8f0">เติม Missing Sentiment</strong> — ใช้ Rule-based จาก gdp_change</li>
    <li><strong style="color:#e2e8f0">Tokenization</strong> — แปลง text เป็น sequence ของตัวเลขด้วย Keras Tokenizer</li>
    <li><strong style="color:#e2e8f0">Padding</strong> — ทำให้ทุก sequence ยาวเท่ากัน (MAX_LEN = 50)</li>
    <li><strong style="color:#e2e8f0">One-hot Encoding</strong> — แปลง label เป็น categorical vector</li>
    <li><strong style="color:#e2e8f0">แบ่งข้อมูล</strong> — Train 70% / Validation 15% / Test 15%</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Theory
st.markdown("""
<div class="section">
<h3>📖 ทฤษฎี — Bidirectional LSTM</h3>
<p>
LSTM (Long Short-Term Memory) เป็น Recurrent Neural Network ที่ออกแบบมาเพื่อจดจำ
ความสัมพันธ์ระยะยาวในข้อมูล sequence เช่น ข้อความ โดยมี 3 gate ได้แก่
Input Gate, Forget Gate, และ Output Gate เพื่อควบคุมการไหลของข้อมูล
</p>
<p>
Bidirectional LSTM อ่านข้อความทั้งไปข้างหน้าและย้อนกลับ ทำให้เข้าใจ context ของแต่ละคำ
ได้ดีกว่า LSTM ทิศทางเดียว
</p>
</div>
""", unsafe_allow_html=True)

# Architecture
st.markdown("### 🏗️ โครงสร้างโมเดล")
layers = [
    ("Embedding Layer", f"input_dim=5000, output_dim=64, input_length=50", "แปลง word index เป็น dense vector ขนาด 64 มิติ"),
    ("Bidirectional LSTM (1)", "units=64, return_sequences=True, dropout=0.2", "อ่านข้อความ 2 ทิศทาง ส่งต่อ sequence ทั้งหมด"),
    ("LSTM (2)", "units=32, dropout=0.2", "สรุป context จาก sequence เป็น vector เดียว"),
    ("Dense", "units=32, activation='relu'", "เรียนรู้ pattern ที่ซับซ้อน"),
    ("Dropout", "rate=0.3", "ป้องกัน overfitting"),
    ("Output (Dense)", "units=3, activation='softmax'", "ทำนาย probability ของ 3 class"),
]
for name, params, desc in layers:
    st.markdown(f"""
    <div class="layer-box">
        <h4>{name} &nbsp;<code>{params}</code></h4>
        <p>{desc}</p>
    </div>
    """, unsafe_allow_html=True)

# Training
st.markdown("""
<div class="section" style="margin-top:1rem">
<h3>🛠️ ขั้นตอนการ Train โมเดล</h3>
<ul>
    <li>Optimizer: <code>Adam</code> — ปรับ learning rate อัตโนมัติ</li>
    <li>Loss Function: <code>Categorical Crossentropy</code> — สำหรับ multi-class</li>
    <li>Epochs: สูงสุด 30 epoch พร้อม <code>EarlyStopping</code> (patience=5)</li>
    <li>Batch Size: 32</li>
    <li>ประเมินผลด้วย Accuracy, Precision, Recall, F1-Score</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Images
st.markdown("### 📊 ผลการประเมินโมเดล")
col1, col2 = st.columns(2)
with col1:
    try:
        st.image("models/images/confusion_matrix_lstm.png", caption="Confusion Matrix", use_column_width=True)
    except:
        st.info("ยังไม่พบรูป — รัน train_models_v2.py ก่อน")
with col2:
    try:
        st.image("models/images/lstm_training_history.png", caption="Training History", use_column_width=True)
    except:
        st.info("ยังไม่พบรูป — รัน train_models_v2.py ก่อน")

# References
st.markdown("""
<div class="section">
<h3>📚 แหล่งอ้างอิง</h3>
<ul>
    <li><a class="ref-link" href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM" target="_blank">TensorFlow Keras LSTM Documentation</a></li>
    <li><a class="ref-link" href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/" target="_blank">Understanding LSTM Networks — Colah's Blog</a></li>
    <li>Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8).</li>
    <li>Schuster, M. & Paliwal, K.K. (1997). Bidirectional Recurrent Neural Networks. IEEE Transactions.</li>
</ul>
</div>
""", unsafe_allow_html=True)
