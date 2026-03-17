import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Analyze News Sentiment", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans Thai', sans-serif; }
    .result-positive {
        background: linear-gradient(135deg, #1a3a2a, #22543d);
        border: 1px solid #68d391; border-radius: 12px;
        padding: 2rem; text-align: center;
    }
    .result-neutral {
        background: linear-gradient(135deg, #1a1f2e, #2d3748);
        border: 1px solid #63b3ed; border-radius: 12px;
        padding: 2rem; text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, #3a1a1a, #742a2a);
        border: 1px solid #fc8181; border-radius: 12px;
        padding: 2rem; text-align: center;
    }
    .result-positive h2 { color: #68d391; font-size: 2rem; margin: 0.3rem 0; }
    .result-neutral h2  { color: #63b3ed; font-size: 2rem; margin: 0.3rem 0; }
    .result-negative h2 { color: #fc8181; font-size: 2rem; margin: 0.3rem 0; }
    .result-positive p, .result-neutral p, .result-negative p { color: #a0aec0; margin: 0.3rem 0; }
    .prob-bar-wrap { background: #0f1117; border-radius: 8px; overflow: hidden; height: 10px; margin: 4px 0; }
    .prob-bar { height: 10px; border-radius: 8px; }
    .example-chip {
        display: inline-block; background: #1a1f2e; border: 1px solid #2d3748;
        border-radius: 8px; padding: 0.4rem 0.8rem; margin: 0.3rem;
        cursor: pointer; color: #a0aec0; font-size: 0.85rem;
    }
    h1 { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🧠 ทดสอบ Neural Network (LSTM)")
st.markdown("พิมพ์หัวข้อข่าวเศรษฐกิจ แล้วโมเดลจะวิเคราะห์ว่าเป็นข่าว **Positive / Neutral / Negative**")
st.markdown("---")

# Load model
@st.cache_resource
def load_lstm_model():
    model = load_model("models/lstm_model.h5")
    tokenizer = joblib.load("models/tokenizer.pkl")
    config = joblib.load("models/lstm_config.pkl")
    return model, tokenizer, config

try:
    lstm_model, tokenizer, config = load_lstm_model()
    MAX_LEN = config["MAX_LEN"]
    model_loaded = True
except Exception as e:
    st.error(f"❌ โหลดโมเดลไม่ได้: {e}")
    model_loaded = False

if model_loaded:

    # ตัวอย่างข่าว
    st.markdown("**💡 ตัวอย่างข่าว (คลิกเพื่อใช้):**")
    examples = [
        "China GDP growth surges to 8% driven by strong exports",
        "Japan economy contracts amid weak consumer spending",
        "Thailand GDP remains stable at 3.5% in 2021",
        "Indonesia faces economic slowdown due to global uncertainty",
        "Vietnam emerges as top investment destination in Southeast Asia",
    ]

    selected_example = None
    cols = st.columns(len(examples))
    for i, (col, ex) in enumerate(zip(cols, examples)):
        with col:
            if st.button(f"📰 ตัวอย่าง {i+1}", key=f"ex_{i}", use_container_width=True):
                selected_example = ex

    # Input
    default_text = selected_example if selected_example else ""
    headline = st.text_area(
        "📝 พิมพ์หัวข้อข่าวภาษาอังกฤษ:",
        value=default_text,
        height=100,
        placeholder="เช่น: China GDP growth surges to 8% driven by strong exports"
    )

    if st.button("🔮 วิเคราะห์ Sentiment", use_container_width=True, type="primary"):
        if not headline.strip():
            st.warning("⚠️ กรุณากรอกหัวข้อข่าวก่อน")
        else:
            # Predict
            seq = tokenizer.texts_to_sequences([headline])
            padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
            proba = lstm_model.predict(padded, verbose=0)[0]
            pred_class = np.argmax(proba)

            labels = ["Negative", "Neutral", "Positive"]
            colors = ["#fc8181", "#63b3ed", "#68d391"]
            css_classes = ["result-negative", "result-neutral", "result-positive"]
            emojis = ["📉", "➡️", "📈"]

            label = labels[pred_class]
            emoji = emojis[pred_class]
            css = css_classes[pred_class]

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="{css}">
                <div style="font-size:3rem">{emoji}</div>
                <h2>{label}</h2>
                <p>"{headline}"</p>
                <p style="margin-top:0.5rem">ความมั่นใจ: <strong>{proba[pred_class]*100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**📊 Probability ของแต่ละ Sentiment:**")

            for i, (lbl, prob, color) in enumerate(zip(labels, proba, colors)):
                col_l, col_b, col_r = st.columns([2, 6, 2])
                with col_l:
                    st.markdown(f"<span style='color:{color};font-weight:600'>{lbl}</span>", unsafe_allow_html=True)
                with col_b:
                    st.progress(float(prob))
                with col_r:
                    st.markdown(f"<span style='color:#a0aec0'>{prob*100:.1f}%</span>", unsafe_allow_html=True)

    # ตัวอย่างข้อมูลจริง
    with st.expander("📰 ตัวอย่างข้อมูลจริงจาก Dataset"):
        try:
            import pandas as pd
            df = pd.read_csv("data/asia_news_cleaned.csv")
            st.dataframe(
                df[["headline", "country", "year", "gdp_change", "sentiment"]].head(10),
                use_container_width=True
            )
        except:
            st.info("ไม่พบไฟล์ data/asia_news_cleaned.csv")
