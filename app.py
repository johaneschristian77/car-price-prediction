import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from Orange.data import Table

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "Automobile_price_cleaned.csv"
MODEL_PATH = ROOT / "Car_Price_Random_Forest.pkcls"


@st.cache_data
def read_csv():
    return pd.read_csv(CSV_PATH)


@st.cache_resource
def read_model():
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


st.title("🚗 Car Price Prediction")
st.write(
    "Masukkan spesifikasi mobil untuk memperoleh estimasi harga "
    "menggunakan model Random Forest yang dibuat di Orange Data Mining."
)

if not CSV_PATH.exists() or not MODEL_PATH.exists():
    st.error(
        "Dataset atau model belum ditemukan. Pastikan file "
        "`Automobile_price_cleaned.csv` dan `Car_Price_Random_Forest.pkcls` "
        "berada di folder utama repository GitHub."
    )
    st.stop()

try:
    df = read_csv()
    model = read_model()
    domain = model.original_domain
except Exception as e:
    st.error(f"Gagal membaca dataset/model: {e}")
    st.info(
        "Pastikan file model berasal dari Orange 3.41 dan requirements.txt "
        "memasang Orange3 versi yang kompatibel."
    )
    st.stop()

features = list(domain.attributes)
if not features:
    st.error("Model tidak memiliki daftar fitur input.")
    st.stop()

st.subheader("Masukkan spesifikasi mobil")
st.caption("Nilai awal numerik diambil dari median dataset. Kamu dapat mengubahnya.")

values = {}
col1, col2 = st.columns(2)

for i, var in enumerate(features):
    col = col1 if i % 2 == 0 else col2
    with col:
        key = var.name
        label = key.replace("-", " ").title()

        if var.is_discrete:
            options = list(var.values)
            if not options:
                st.warning(f"Pilihan untuk {label} tidak tersedia.")
                continue
            values[key] = st.selectbox(label, options, key=f"field_{key}")

        elif var.is_continuous:
            if key in df.columns:
                series = pd.to_numeric(df[key], errors="coerce").dropna()
            else:
                series = pd.Series(dtype=float)

            if series.empty:
                low, high, default = 0.0, 100.0, 0.0
            else:
                low = float(series.min())
                high = float(series.max())
                default = float(series.median())

            if low == high:
                high = low + 1.0

            values[key] = st.number_input(
                label,
                min_value=low,
                max_value=high,
                value=min(max(default, low), high),
                step=0.1 if high - low < 100 else 1.0,
                key=f"field_{key}",
            )
        else:
            st.warning(f"Jenis fitur {label} belum didukung.")
            values[key] = ""

if st.button("Prediksi Harga", type="primary", use_container_width=True):
    try:
        # Orange expects values in the exact order of the model's original domain:
        # attributes, class variables, then meta variables.
        row = [values[var.name] for var in domain.attributes]

        for class_var in domain.class_vars:
            row.append(np.nan)

        for meta_var in domain.metas:
            row.append("")

        input_table = Table.from_list(domain, [row])
        prediction = model(input_table)
        estimated_price = float(np.asarray(prediction).reshape(-1)[0])

        st.success("Prediksi berhasil dibuat.")
        st.metric("Estimasi Harga Mobil", f"{estimated_price:,.2f}")
        st.caption(
            "Hasil ini merupakan estimasi model berdasarkan dataset, "
            "bukan harga jual pasti."
        )
    except Exception as e:
        st.error(f"Prediksi gagal: {e}")
        st.info(
            "Kirimkan screenshot pesan error ini agar kita dapat "
            "menyesuaikan input dengan domain model Orange."
        )

with st.expander("Tentang model"):
    st.write(
        "Model yang digunakan adalah Random Forest dari Orange Data Mining. "
        "Evaluasi sebelumnya menggunakan 5-fold Cross Validation."
    )
