import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Konfigurasi halaman
st.set_page_config(page_title="LMS Mahasiswa", layout="wide")

# Load dataset mahasiswa
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset_mahasiswa_812.csv")
        
        # Konversi status akademik ke numerik
        ipk_mapping = {
            'IPK < 2.5': 0,
            'IPK 2.5–3.0': 1,
            'IPK > 3.0': 2
        }
        df['status_akademik_numerik'] = df['status_akademik_terakhir'].map(ipk_mapping)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# 🔐 Login dari CSV
st.sidebar.header("🔐 Login Mahasiswa")
nama_list = df["Nama"].unique().tolist()
nama = st.sidebar.selectbox("Pilih Nama Mahasiswa", nama_list)
nim_input = st.sidebar.text_input("Masukkan NIM Mahasiswa")

# Validasi Nama & NIM
valid_mahasiswa = df[df["Nama"] == nama]
if not valid_mahasiswa.empty:
    nim_terdaftar = str(valid_mahasiswa.iloc[0]["ID Mahasiswa"])
    login_berhasil = (nim_input == nim_terdaftar)
else:
    login_berhasil = False

# ✅ Jika login berhasil
if login_berhasil:
    st.title(f"🎓 LMS Mahasiswa - {nama}")
    menu = st.sidebar.radio("Navigasi", ["Beranda", "Materi", "Tugas", "Prediksi Dropout"])

    if menu == "Beranda":
        st.subheader(f"👋 Selamat Datang, {nama}!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Status Login", "Aktif")
        
        # Ambil data mahasiswa yang login
        mhs_data = valid_mahasiswa.iloc[0]
        
        # Tampilkan IPK sesuai mapping
        ipk_status = mhs_data["status_akademik_terakhir"]
        col2.metric("Status Akademik", ipk_status)
        
        # Hitung progress berdasarkan materi selesai
        progress = mhs_data["materi_selesai"] / 100
        col3.metric("Materi Selesai", f"{mhs_data['materi_selesai']}%")
        st.progress(progress)

    elif menu == "Materi":
        st.subheader("📘 Materi Pembelajaran")
        with st.expander("Modul 1 - Dasar Pemrograman"):
            st.markdown("📄 Pengantar Python dan Algoritma")
        with st.expander("Modul 2 - Analisis Data"):
            st.markdown("🧠 Statistika Dasar dan Visualisasi Data")
        with st.expander("Modul 3 - Machine Learning"):
            st.markdown("📊 Model Prediktif dan Evaluasi")

    elif menu == "Tugas":
        st.subheader("📝 Daftar Tugas")
        tugas_data = pd.DataFrame({
            "Judul": ["Tugas 1 - Analisis Eksplorasi", "Tugas 2 - Model Prediksi", "Tugas 3 - Presentasi"],
            "Status": ["✅ Selesai", "❌ Belum", "❌ Belum"],
            "Deadline": ["2025-06-15", "2025-06-25", "2025-07-01"]
        })
        st.table(tugas_data)

        st.markdown("### 📎 Upload Tugas")
        uploaded = st.file_uploader("Upload file tugas (.pdf/.docx/.ipynb)", type=["pdf", "docx", "ipynb"])
        if uploaded:
            st.success(f"File '{uploaded.name}' berhasil diunggah!")

    elif menu == "Prediksi Dropout":
        st.subheader("📊 Hasil Prediksi Dropout")
        mahasiswa = df[df["Nama"] == nama]

        if not mahasiswa.empty:
            X = mahasiswa.drop(columns=["ID Mahasiswa", "Nama", "NIM", "dropout"])
            proba = model.predict_proba(X)[0][1]
            pred = model.predict(X)[0]

            st.metric("Probabilitas Dropout", f"{proba:.2%}")
            if pred == 1:
                st.error("❌ Mahasiswa ini diprediksi berisiko dropout.")
            else:
                st.success("✅ Mahasiswa ini diprediksi tidak dropout.")

            st.markdown("---")
            st.subheader("📈 Visualisasi SHAP (Waterfall Plot)")

            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            shap.plots.waterfall(shap_values[0])
            st.pyplot(plt.gcf())
            plt.clf()

            st.markdown("---")
            st.subheader("📊 Distribusi Dropout Keseluruhan")
            dropout_counts = df['dropout'].value_counts()
            labels = ['Tidak Dropout', 'Dropout']
            colors = ['#28a745', '#dc3545']

            fig, ax = plt.subplots()
            ax.pie(dropout_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')
            st.pyplot(fig)

else:
    st.title("🎓 LMS Mahasiswa")
    st.warning("Nama atau NIM tidak cocok. Silakan login kembali.")
