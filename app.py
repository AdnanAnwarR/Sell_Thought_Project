import json, datetime as dt
from pathlib import Path
import joblib, pandas as pd, streamlit as st

BASE = Path(__file__).parent
st.set_page_config(page_title="Inventory Sell-Through Predictor", page_icon="📦", layout="wide")

@st.cache_resource
def load():
    return joblib.load(BASE / "model" / "model.joblib"), json.load(open(BASE / "model" / "meta.json")), json.load(open(BASE / "model" / "results.json"))
model, meta, res = load()
thr = meta["threshold"]

st.title("📦 Inventory Sell-Through Predictor")
st.caption("Memprediksi peluang sebuah item inventori terjual (dalam jendela 60 hari) untuk membantu perencanaan stok.")
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi", "📊 Insight Data", "🧪 Model & MLOps"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        stock_date = st.date_input("Tanggal stok dibuat", value=dt.date.fromisoformat(meta["cutoff"]))
        cat = st.selectbox("Kategori produk", meta["categories"])
        dept = st.selectbox("Departemen", meta["departments"])
        brand = st.selectbox("Brand", meta["top_brands"] + ["(brand lain)"])
    with c2:
        dc = st.selectbox("Distribution center", meta["dcs"])
        price = st.number_input("Harga jual (USD)", 1.0, 1000.0, 50.0, 1.0)
        default_cost = round(price * meta["avg_cost_ratio_by_cat"].get(cat, 0.5), 2)
        cost = st.number_input("Biaya/cost (USD)", 0.1, 1000.0, float(default_cost), 0.5,
                               help="Default = rata-rata rasio cost/harga kategori tersebut.")
    if cost >= price:
        st.error("Cost harus lebih kecil dari harga jual.")
    elif st.button("Prediksi", type="primary"):
        row = pd.DataFrame([{"created_at": pd.Timestamp(stock_date, tz="UTC"), "cost": cost, "product_category": cat,
                             "product_brand": None if brand == "(brand lain)" else brand, "product_retail_price": price,
                             "product_department": dept, "product_distribution_center_id": dc}])
        p = float(model.predict_proba(row)[0, 1])
        a, b, c = st.columns(3)
        a.metric("Peluang terjual", f"{p:.1%}", f"{(p - meta['base_rate']) * 100:+.1f} poin vs rata-rata")
        b.metric("Margin per unit", f"${price - cost:,.2f}", f"{(price - cost) / price:.0%}")
        c.metric("Kategori risiko", "Risiko dead stock tinggi" if p < thr else "Cenderung terjual")
        st.progress(min(max(p, 0.0), 1.0))
        if stock_date > dt.date.fromisoformat(meta["cutoff"]):
            st.warning("Tanggal melewati rentang data latih. Hasil uji temporal menunjukkan model tidak dapat "
                       "mengekstrapolasi tren waktu — anggap prediksi ini hanya indikatif.")
        st.caption(f"Threshold klasifikasi {thr:.2f} (dipilih memaksimalkan F1 pada holdout, recall tinggi).")

with tab2:
    st.subheader("Insight utama")
    st.markdown("- Stok 2018–2019 100% terjual (patahan di Jan 2020, tampak seperti artefak data); sejak 2020 sell-through naik seiring waktu pembuatan stok; **kategori, brand, harga, DC tidak membedakan** peluang terjual.\n"
                "- Waktu terjual selalu 0–60 hari (seragam) → item yang belum terjual setelah 60 hari tidak terjual di data.\n"
                "- Item stok terakhir (<60 hari sebelum akhir data) dibuang karena *right-censored*.")
    figs = sorted((BASE / "figures").glob("f[1-5]*.png"))
    cols = st.columns(2)
    for i, f in enumerate(figs):
        cols[i % 2].image(str(f), use_container_width=True)

with tab3:
    fh = res["final_holdout"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC (holdout)", f"{fh['auc']:.3f}"); m2.metric("PR-AUC", f"{fh['pr_auc']:.3f}")
    m3.metric("Recall @thr", f"{fh['recall']:.2f}"); m4.metric("Precision @thr", f"{fh['precision']:.2f}")
    st.image(str(BASE / "figures" / "f6_model_eval.png"), use_container_width=True)
    st.markdown(f"""**Uji temporal:** AUC {res['temporal_holdout_auc']:.3f} saat model dilatih sampai Mei 2023 dan diuji sesudahnya → **drift/ekstrapolasi** adalah risiko utama.

**Praktik MLOps:** satu pipeline (FeatureBuilder + encoder + model) disimpan dalam `model.joblib` sehingga fitur training = serving;
Docker untuk reproduksi; monitoring yang disarankan: rata-rata probabilitas prediksi vs sell-through aktual per bulan, dan retrain bila AUC bulanan < 0.65.""")
