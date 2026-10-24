# 📦 Inventory Sell-Through Prediction

Proyek Data Science / Machine Learning untuk memprediksi **apakah sebuah unit inventori akan terjual dalam jendela 60 hari** sejak stok dibuat (klasifikasi biner), lengkap dengan aplikasi **Streamlit** dan dukungan **Docker**.

Proyek ini dibuat sebagai take-home test DSML: mulai dari eksplorasi data, rekayasa fitur, pemodelan, evaluasi, sampai deployment aplikasi interaktif.

---

## 1. Latar Belakang & Tujuan

Bisnis retail menghadapi risiko **dead stock** — barang yang menumpuk di gudang dan tidak kunjung terjual. Jika tim inventory bisa memperkirakan peluang sebuah item terjual sejak stok dibuat, keputusan seperti diskon, redistribusi stok, atau pengurangan order berikutnya bisa diambil lebih awal.

**Tujuan:**

- Memprediksi probabilitas sell-through (terjual dalam 60 hari) untuk satu item inventori.
- Mengidentifikasi fitur yang benar-benar memiliki sinyal prediktif.
- Menyediakan aplikasi yang bisa dipakai tim non-teknis untuk simulasi prediksi.
- Menyiapkan artefak MLOps sederhana: pipeline tersimpan, metrik, dan Docker.

**Target variabel:** `is_sold` = 1 jika `sold_at` terisi, 0 jika tidak.

---

## 2. Dataset

Data yang dipakai adalah `inventory_items.csv` dan **sudah disertakan di repo ini** (± 89,6 MiB / 94 MB). Karena ukurannya cukup besar, clone pertama akan mengunduh file ini. Dataset berisi ± 490 ribu baris unit inventori dengan rentang data 2018–2024.

Kolom utama yang dipakai:

| Kolom | Deskripsi |
|---|---|
| `id` | ID unik baris inventori |
| `product_id` | ID produk (dipakai untuk grouping agar tidak bocor antar train/test) |
| `created_at` | Waktu stok dibuat |
| `sold_at` | Waktu terjual (`NaN` jika tidak terjual) |
| `cost` | Biaya perolehan per unit |
| `product_retail_price` | Harga jual |
| `product_category` | Kategori produk |
| `product_department` | Departemen (Men/Women) |
| `product_brand` | Brand produk |
| `product_distribution_center_id` | ID distribution center |

**Lokasi data:** file `inventory_items.csv` sudah berada di root folder proyek (satu folder dengan `app.py` dan notebook). Jika mengunduh repo sebagai ZIP, pastikan file tersebut terekstrak di root. Training membutuhkan file ini; aplikasi **tidak** membutuhkannya karena hanya memakai model yang sudah dilatih.

---

## 3. Pendekatan

### 3.1 Pembersihan & penanganan censoring

- Duplikat `id` dibuang.
- `is_sold` dibentuk dari `sold_at.notna()`.
- Item yang dibuat **kurang dari 60 hari sebelum akhir data** dibuang karena *right-censored*: belum cukup waktu untuk bisa terjual, bukan berarti tidak akan terjual. Batas censoring ada di `features.CENSOR_CUTOFF` (`2023-11-22`).

### 3.2 Rekayasa fitur

Dibangun di `FeatureBuilder` (`features.py`) supaya logika fitur **identik antara training dan serving** (menghindari training-serving skew):

- **Waktu:** `t_days` (hari sejak `2018-11-08`), `month_sin`, `month_cos`, `dow`.
- **Harga & margin:** `log_price`, `log_cost`, `margin_abs`, `margin_pct`.
- **Kategorikal:** kategori, departemen, brand, distribution center. Brand langka (< 50 kemunculan) digabung menjadi `OTHER`.
- `sold_at` **tidak** dipakai sebagai fitur → tidak ada data leakage.

### 3.3 Split & validasi

- Holdout **20%** dengan `GroupShuffleSplit` berdasarkan `product_id`, agar produk yang sama tidak muncul di train dan test sekaligus.
- Cross-validation memakai `GroupKFold(3)`.
- Uji ekstrapolasi temporal: latih data sebelum Juni 2023, uji sesudahnya.

### 3.4 Model

Kandidat baseline: Logistic Regression, Decision Tree, Random Forest, HistGradientBoosting. Model final adalah **HistGradientBoostingClassifier** yang di-tuning dengan `RandomizedSearchCV`, lalu di-refit pada seluruh data modelling dan disimpan sebagai satu pipeline (`FeatureBuilder` + preprocessing + model) di `model/model.joblib`.

Sebagai pembanding, `dl.py` melatih neural network sederhana (Keras/TensorFlow).

### 3.5 Threshold

Threshold klasifikasi dipilih dengan memaksimalkan **F1 pada holdout** (hasil: `0.29`), sehingga model condong ke recall tinggi — cocok untuk skenario di mana melewatkan item berisiko dead stock lebih mahal daripada false alarm.

---

## 4. Hasil

Metrik pada holdout (20%):

| Metrik | Nilai |
|---|---|
| ROC-AUC | **0.699** |
| PR-AUC | **0.561** |
| Log loss | 0.591 |
| Brier score | 0.203 |
| F1 @ threshold 0.29 | 0.579 |
| Precision @ threshold | 0.463 |
| Recall @ threshold | 0.773 |
| Base rate (proporsi terjual) | 35.7% |

**Temuan penting:**

- **Waktu pembuatan stok adalah satu-satunya fitur yang punya sinyal kuat.** Ablation: hanya `t_days` → AUC 0.696; semua fitur → 0.699.
- Harga, kategori, brand, departemen, dan DC **hampir tidak membedakan** peluang terjual (AUC ≈ 0.50 bila hanya fitur-fitur ini).
- Uji temporal (latih < Jun 2023, uji ≥ Jun 2023) menghasilkan **AUC ≈ 0.498** → model **tidak bisa mengekstrapolasi tren waktu**. Ini risiko drift utama; prediksi untuk tanggal di luar rentang data latih hanya bersifat indikatif (aplikasi menampilkan peringatan untuk tanggal setelah cutoff).
- 100% stok 2018–2019 terjual, lalu ada patahan di Jan 2020 (kemungkinan artefak data).

Visualisasi ada di folder `figures/`:

| File | Isi |
|---|---|
| `f1_sold_by_quarter.png` | % item terjual per kuartal pembuatan stok |
| `f2_days_to_sell.png` | Distribusi waktu terjual (0–60 hari) |
| `f3_sold_by_category.png` | Sell-through per kategori (hampir flat) |
| `f4_margin_price.png` | Distribusi margin & harga vs status terjual |
| `f5_monthly_trend.png` | Tren sell-through bulanan |
| `f6_model_eval.png` | ROC, ablasi fitur, permutation importance |
| `f7_calibration.png` | Kalibrasi probabilitas (decile) |

---

## 5. Struktur Proyek

```
sellthrough_project/
├── inventory_items.csv         # Dataset mentah (± 89,6 MiB, sudah disertakan)
├── app.py                      # Aplikasi Streamlit (3 tab: Prediksi, Insight, Model & MLOps)
├── features.py                 # FeatureBuilder + konstanta fitur (dipakai training & app)
├── prep.py                     # Load & cleaning data, handling censoring
├── train.py                    # Baseline model + holdout split
├── tune.py                     # Ablasi, uji temporal, tuning, model final, artefak
├── dl.py                       # Eksperimen deep learning (Keras)
├── figs.py                     # Generate semua figur ke figures/
├── test_app.py                 # Smoke test aplikasi dengan streamlit.testing
├── THT_Inventory_SellThrough.ipynb  # Notebook dokumentasi lengkap (EDA → model → evaluasi)
├── requirements.txt
├── Dockerfile
├── model/
│   ├── model.joblib            # Pipeline final siap deploy
│   ├── meta.json               # Metadata untuk UI (threshold, kategori, brand, dll.)
│   ├── results.json            # Metrik, ablasi, importance, parameter terbaik
│   ├── baseline_results.csv
│   ├── tuning_results.csv
│   ├── calibration.csv
│   ├── dl_results.json
│   ├── holdout_proba.npy / holdout_y.npy
│   └── split_idx.joblib
└── figures/                    # f1–f7.png
```

---

## 6. Setup & Menjalankan Aplikasi

### Prasyarat

- **Python 3.10–3.14** (Docker memakai 3.12)
- `pip`
- Opsional: Docker Desktop

> ⚠️ **Penting:** `model/model.joblib` dibuat dengan **scikit-learn 1.8.0**. Jika environment memakai scikit-learn versi lain, model bisa gagal di-load (mis. `ModuleNotFoundError: No module named '_loss'`) atau memberi hasil tidak valid. Selalu install dari `requirements.txt` di environment terpisah (virtualenv).

### 6.1 Menjalankan secara lokal (direkomendasikan)

**Windows (PowerShell / CMD):**

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Buka **http://localhost:8501** di browser.

Jika tidak ingin mengaktifkan venv, perintah bisa dijalankan langsung, misalnya di Windows:

```bat
.venv\Scripts\python -m streamlit run app.py
```

### 6.2 Menjalankan dengan Docker

```bash
docker build -t sellthrough .
docker run -p 8501:8501 sellthrough
```

Buka **http://localhost:8501**. Image hanya menyalin `app.py`, `features.py`, `prep.py`, artefak model yang dibutuhkan, dan `figures/` — data mentah tidak ikut.

### 6.3 Menjalankan smoke test aplikasi

```bash
python test_app.py
```

Test ini menjalankan `app.py` lewat `streamlit.testing.v1.AppTest`, memastikan tidak ada exception saat halaman dimuat maupun saat tombol **Prediksi** diklik, lalu mencetak nilai metrik.

---

## 7. Menjalankan Ulang Training

Training membutuhkan `inventory_items.csv`, yang sudah tersedia di root proyek. Untuk `dl.py` dibutuhkan TensorFlow (tidak termasuk di `requirements.txt` aplikasi).

```bash
# urutan yang disarankan
python train.py    # baseline + split index
python tune.py     # ablasi, uji temporal, tuning, model final -> model/model.joblib
python dl.py       # opsional: eksperimen deep learning (butuh tensorflow)
python figs.py     # generate figures/f1..f7
```

Catatan:

- `train.py` menulis `model/split_idx.joblib` yang dipakai `tune.py` dan `dl.py`.
- `tune.py` menulis ulang `model/model.joblib`, `meta.json`, dan `results.json`. Setelah training ulang, jalankan `python test_app.py` untuk memastikan aplikasi tetap berjalan.
- Training di dataset penuh butuh RAM dan waktu cukup besar (ratusan ribu baris). `dl.py` sengaja memakai sampel 80.000 baris agar hemat memori.

### Menjalankan notebook

```bash
jupyter notebook THT_Inventory_SellThrough.ipynb
```

Notebook bersifat mandiri (EDA, preprocessing, model, evaluasi) selama `inventory_items.csv` tersedia di folder yang sama.

---

## 8. Menggunakan Aplikasi

Aplikasi memiliki tiga tab:

1. **🔮 Prediksi** — isi tanggal stok, kategori, departemen, brand, distribution center, harga, dan cost. Output: peluang terjual, margin per unit, kategori risiko (dead stock tinggi vs cenderung terjual), serta peringatan bila tanggal berada di luar rentang data latih.
2. **📊 Insight Data** — ringkasan temuan utama dan figur EDA.
3. **🧪 Model & MLOps** — metrik holdout, ROC/ablasi/importance, hasil uji temporal, dan praktik MLOps yang diterapkan.

---

## 9. MLOps & Monitoring

- **Satu pipeline untuk training dan serving** — `FeatureBuilder` + encoder + model disimpan dalam `model/model.joblib`, sehingga fitur saat training sama dengan saat serving.
- **Metadata terpusat** — `model/meta.json` menyediakan threshold, kategori, brand populer, rasio cost/harga rata-rata per kategori, dan cutoff untuk UI.
- **Reproducible** — versi library dipin di `requirements.txt`, aplikasi dapat dibangun lewat Docker.
- **Monitoring yang disarankan:**
  - Bandingkan rata-rata probabilitas prediksi dengan sell-through aktual **per bulan** (deteksi drift).
  - Pantau AUC bulanan; **retrain bila AUC < 0.65**.
  - Waspadai prediksi untuk stok yang dibuat setelah `cutoff` data latih, karena model tidak mampu mengekstrapolasi tren waktu.

---

## 10. Keterbatasan

- Sinyal prediktif hampir seluruhnya berasal dari **waktu pembuatan stok**, bukan atribut produk. Model lebih tepat dipandang sebagai estimator tren sell-through, bukan pembeda antar produk.
- Performa temporal rendah (AUC ≈ 0.50) menunjukkan model tidak robust terhadap perubahan rezim/data di masa depan.
- Data 2018–2019 dengan sell-through 100% dan patahan Jan 2020 perlu diverifikasi ke sumber data sebelum dipakai untuk keputusan bisnis.
- Model belum dikalibrasi ulang per periode; kalibrasi keseluruhan ditampilkan di `f7_calibration.png`.
