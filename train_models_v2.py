import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# ============================================================
# CONFIG — แก้ path ตรงนี้ให้ตรงกับที่วางไฟล์ CSV
# ============================================================
GDP_CSV   = "data/asia_gdp_cleaned.csv"
NEWS_CSV  = "data/asia_news_cleaned.csv"
MODEL_DIR = "models"
IMG_DIR   = os.path.join(MODEL_DIR, "images")

# สร้าง folder ถ้ายังไม่มี
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR,   exist_ok=True)

# ============================================================
print("\n" + "=" * 60)
print("  โมเดลที่ 1 : ML Ensemble — GDP Growth Prediction")
print("=" * 60)
# ============================================================

# ── 1.1 โหลดข้อมูล ──────────────────────────────────────────
df_gdp = pd.read_csv(GDP_CSV)
print(f"[โหลดข้อมูล] Shape: {df_gdp.shape}")
print(df_gdp.head(3).to_string())

# ── 1.2 เตรียม Features และ Target ──────────────────────────
year_cols   = [str(y) for y in range(2013, 2021)]  # features = GDP ปี 2013-2020
target_year = "2021"                                # predict ปี 2021

X = df_gdp[year_cols].copy()

# engineered features
X["mean_gdp"]    = X[year_cols].mean(axis=1)
X["std_gdp"]     = X[year_cols].std(axis=1)
X["trend"]       = X["2020"] - X["2013"]
X["last_2yr_avg"]= (X["2019"] + X["2020"]) / 2

# target: 1 = GDP ขึ้น (>= 0), 0 = GDP ลง (< 0)
y = (df_gdp[target_year] >= df_gdp[year_cols].mean(axis=1)).astype(int)

print(f"\n[Target distribution]\n{y.value_counts().to_string()}")
print(f"GDP ขึ้น (1): {y.sum()} | GDP ลง (0): {(y==0).sum()}")

# ── 1.3 Train/Test Split + Scale ────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_test_sc    = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── 1.4 Cross-Validation ของ Base Learners ──────────────────
print("\n[Cross-Validation ของ Base Learners]")
base_models = [
    ("Random Forest",      RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ("XGBoost",            XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                         use_label_encoder=False, eval_metric="logloss", random_state=42)),
    ("Gradient Boosting",  GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)),
]
for name, mdl in base_models:
    scores = cross_val_score(mdl, X_train_sc, y_train, cv=5, scoring="accuracy")
    print(f"  {name:25s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# ── 1.5 Train Stacking Ensemble ─────────────────────────────
print("\n[Train Stacking Ensemble] กำลัง train...")
estimators = [
    ("rf",  RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ("xgb", XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                          use_label_encoder=False, eval_metric="logloss", random_state=42)),
    ("gb",  GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)),
]
ensemble_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(random_state=42),
    cv=5,
    passthrough=True
)
ensemble_model.fit(X_train_sc, y_train)
print("✅ Train เสร็จแล้ว!")

# ── 1.6 ประเมินผล ────────────────────────────────────────────
y_pred = ensemble_model.predict(X_test_sc)
acc    = accuracy_score(y_test, y_pred)
print(f"\n[ผลการประเมิน] Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(classification_report(y_test, y_pred, target_names=["GDP ลง (0)", "GDP ขึ้น (1)"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["GDP ลง", "GDP ขึ้น"],
            yticklabels=["GDP ลง", "GDP ขึ้น"])
plt.title("Confusion Matrix — ML Ensemble (GDP)")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "confusion_matrix_ml.png"), dpi=150)
plt.close()

# Feature Importance
rf_mdl   = ensemble_model.named_estimators_["rf"]
feat_imp = pd.Series(rf_mdl.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 4))
feat_imp.plot(kind="bar", color="steelblue")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Feature"); plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "feature_importance_ml.png"), dpi=150)
plt.close()

# ── 1.7 บันทึกโมเดล ─────────────────────────────────────────
joblib.dump(ensemble_model,      os.path.join(MODEL_DIR, "ensemble_model.pkl"))
joblib.dump(scaler,              os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(list(X.columns),     os.path.join(MODEL_DIR, "feature_columns.pkl"))
print("\n✅ บันทึกโมเดล ML สำเร็จ:")
print("   models/ensemble_model.pkl")
print("   models/scaler.pkl")
print("   models/feature_columns.pkl")


# ============================================================
print("\n" + "=" * 60)
print("  โมเดลที่ 2 : Neural Network LSTM — Sentiment Classification")
print("=" * 60)
# ============================================================

# ── 2.1 โหลดข้อมูล ──────────────────────────────────────────
df_news = pd.read_csv(NEWS_CSV)
print(f"[โหลดข้อมูล] Shape: {df_news.shape}")
print(df_news[["headline", "sentiment", "sentiment_label"]].head(3).to_string())

# Sentiment distribution
print(f"\n[Sentiment distribution]\n{df_news['sentiment'].value_counts().to_string()}")

# ── 2.2 เตรียม Text Data ─────────────────────────────────────
MAX_WORDS     = 5000
MAX_LEN       = 50
EMBEDDING_DIM = 64
NUM_CLASSES   = 3

texts  = df_news["headline"].astype(str).values
labels = df_news["sentiment_label"].values

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences  = tokenizer.texts_to_sequences(texts)
X_padded   = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
y_cat      = to_categorical(labels, num_classes=NUM_CLASSES)

print(f"\nVocabulary size : {len(tokenizer.word_index)}")
print(f"X shape         : {X_padded.shape}")
print(f"y shape         : {y_cat.shape}")

# ── 2.3 Train/Val/Test Split ─────────────────────────────────
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X_padded, y_cat, test_size=0.3,  random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_tmp,    y_tmp, test_size=0.5,  random_state=42)
print(f"\nTrain: {X_tr.shape[0]} | Val: {X_val.shape[0]} | Test: {X_te.shape[0]}")

# ── 2.4 สร้างโครงสร้าง LSTM ──────────────────────────────────
model_lstm = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM,
              input_length=MAX_LEN, name="embedding"),
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2), name="bilstm_1"),
    LSTM(32, dropout=0.2, name="lstm_2"),
    Dense(32, activation="relu", name="dense_1"),
    Dropout(0.3, name="dropout"),
    Dense(NUM_CLASSES, activation="softmax", name="output"),
])
model_lstm.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model_lstm.summary()

# ── 2.5 Train โมเดล ──────────────────────────────────────────
print("\n[Train LSTM] กำลัง train...")
early_stop = EarlyStopping(monitor="val_loss", patience=5,
                           restore_best_weights=True, verbose=1)
history = model_lstm.fit(
    X_tr, y_tr,
    epochs=30, batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop], verbose=1
)
print("✅ Train เสร็จแล้ว!")

# ── 2.6 ประเมินผล ────────────────────────────────────────────
test_loss, test_acc = model_lstm.evaluate(X_te, y_te, verbose=0)
print(f"\n[ผลการประเมิน] Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

y_pred_prob = model_lstm.predict(X_te)
y_pred      = np.argmax(y_pred_prob, axis=1)
y_true      = np.argmax(y_te, axis=1)
print(classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"]))

# Training History
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"],     label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Model Accuracy"); axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy");      axes[0].legend()
axes[1].plot(history.history["loss"],     label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Model Loss"); axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss");      axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "lstm_training_history.png"), dpi=150)
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Negative","Neutral","Positive"],
            yticklabels=["Negative","Neutral","Positive"])
plt.title("Confusion Matrix — LSTM (Sentiment)")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "confusion_matrix_lstm.png"), dpi=150)
plt.close()

# ── 2.7 บันทึกโมเดล ─────────────────────────────────────────
model_lstm.save(os.path.join(MODEL_DIR, "lstm_model.h5"))
joblib.dump(tokenizer, os.path.join(MODEL_DIR, "tokenizer.pkl"))
joblib.dump({"MAX_WORDS": MAX_WORDS, "MAX_LEN": MAX_LEN,
             "EMBEDDING_DIM": EMBEDDING_DIM, "NUM_CLASSES": NUM_CLASSES},
            os.path.join(MODEL_DIR, "lstm_config.pkl"))

print("\n✅ บันทึกโมเดล LSTM สำเร็จ:")
print("   models/lstm_model.h5")
print("   models/tokenizer.pkl")
print("   models/lstm_config.pkl")

# ── สรุปไฟล์ทั้งหมด ──────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅ เสร็จสมบูรณ์! ไฟล์ทั้งหมดที่ได้:")
print("=" * 60)
for root, dirs, files in os.walk(MODEL_DIR):
    for f in files:
        path = os.path.join(root, f)
        size = os.path.getsize(path) / 1024
        print(f"  {path:50s} ({size:.1f} KB)")
